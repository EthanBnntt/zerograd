"""Distributed ZeroGrad: two GPU workers on XOR.

Runs two workers on the same GPU, each evaluating half the population.
This demonstrates the protocol for a multi-GPU node where each GPU runs
one worker — here we simulate it with two shards on one GPU.

The key property: each worker evaluates its candidates independently and
shares only its 1D loss array.  Whether the workers are on separate GPUs
or the same GPU, the protocol is identical.

For a real multi-GPU setup (e.g. 4× GPU node), just pass all GPU devices
to DistributedZeroGrad — no other changes needed.

    uv run python examples/train_distributed_dual_worker.py [--steps N]
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import optax

from zerograd import (
    DistributedZeroGrad,
    Manifest,
    ManifestEntry,
    ParameterLayout,
    ZeroGrad,
)

# ── XOR model: 2→16→1 MLP ────────────────────────────────────────────────────
INPUT_DIM = 2
HIDDEN_DIM = 16
OUTPUT_DIM = 1


def build_params(key):
    k1, k2 = jax.random.split(key)
    return {
        "w1": jax.random.normal(k1, (INPUT_DIM, HIDDEN_DIM)) * 0.5,
        "b1": jnp.zeros((HIDDEN_DIM,)),
        "w2": jax.random.normal(k2, (HIDDEN_DIM, OUTPUT_DIM)) * 0.5,
        "b2": jnp.zeros((OUTPUT_DIM,)),
    }


def build_manifest():
    return Manifest(version=1, entries=(
        ManifestEntry(("w1",), ParameterLayout.MATRIX, "w1"),
        ManifestEntry(("b1",), ParameterLayout.VECTOR, "b1"),
        ManifestEntry(("w2",), ParameterLayout.MATRIX, "w2"),
        ManifestEntry(("b2",), ParameterLayout.VECTOR, "b2"),
    ))


XOR_X = jnp.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
XOR_Y = jnp.array([[0.], [1.], [1.], [0.]])


def loss_fn(params, candidate, batch, rng):
    x, y = batch
    h = jax.nn.tanh(candidate.linear(params, ("w1",), x))
    h = h + candidate.vector(params, ("b1",))
    logits = candidate.linear(params, ("w2",), h)
    logits = logits + candidate.vector(params, ("b2",))
    return jnp.mean((logits - y) ** 2), None


def accuracy(params):
    h = jax.nn.tanh(XOR_X @ params["w1"]) + params["b1"]
    logits = h @ params["w2"] + params["b2"]
    preds = (logits > 0.5).astype(jnp.float32)
    return float(jnp.mean(preds == XOR_Y))


def main():
    parser = argparse.ArgumentParser(description="Distributed ZeroGrad: dual GPU workers on XOR")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--pop", type=int, default=32)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--sigma", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=1e-2)
    args = parser.parse_args()

    gpu = jax.devices("gpu")[0]
    # Two workers on the same GPU
    devices = [gpu, gpu]
    print(f"Worker 0: {gpu.platform} (id={gpu.id})")
    print(f"Worker 1: {gpu.platform} (id={gpu.id})  [shared GPU]")
    print(f"Population: {args.pop} ({args.pop // 2} per worker)")
    print(f"Steps: {args.steps}\n")

    params = build_params(jax.random.key(0))
    manifest = build_manifest()

    base_opt = ZeroGrad(
        manifest,
        optax.adamw(learning_rate=args.lr, weight_decay=0.0),
        population_size=args.pop,
        rank=args.rank,
        sigma=args.sigma,
        seed=42,
        run_id="xor-dual-gpu",
    )

    dist_opt = DistributedZeroGrad(
        base_opt,
        devices=devices,
        loss_fn=loss_fn,
    )

    state = dist_opt.init(params)
    batch = (XOR_X, XOR_Y)

    for shard in dist_opt.shards:
        print(f"  {shard.name}: {shard.device}")

    print()
    t0 = time.time()
    for step in range(args.steps):
        params, state, metrics = dist_opt.step(state, params, batch)
        if step % 50 == 0 or step == args.steps - 1:
            acc = accuracy(params)
            print(f"  gen {metrics.generation:3d}  "
                  f"loss={metrics.mean_loss:.4f}  "
                  f"acc={acc:.0%}  "
                  f"({(time.time() - t0) / (step + 1):.2f}s/step)")

    print(f"\nFinal accuracy: {accuracy(params):.0%}")
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
