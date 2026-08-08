"""Asymmetric distributed ZeroGrad: handling unequal compute.

When workers have different speeds (slow CPU + fast GPU, or mixed GPU
generations), an even population split wastes the fast device — it finishes
early and idles while the slow device catches up.

This demo shows three approaches:

  1. Even split (naive) — equal candidates per device, fast device idles.
  2. Manual weights — user specifies relative compute weights (e.g. [1, 4]).
  3. Auto-calibration — time each device, set weights from measured speed.

The model is intentionally large enough to show a CPU/GPU speed difference.

    uv run python examples/train_distributed_asymmetric.py [--steps N]
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from zerograd import DistributedZeroGrad, ZeroGrad

# ── Model: 512→512→10 MLP (large enough for GPU to show advantage) ──────────
INPUT_DIM = 512
HIDDEN_DIM = 512
OUTPUT_DIM = 10


class BigMLP(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.l1 = nnx.Linear(INPUT_DIM, HIDDEN_DIM, rngs=rngs)
        self.l2 = nnx.Linear(HIDDEN_DIM, OUTPUT_DIM, rngs=rngs)

    def __call__(self, x):
        return self.l2(nnx.relu(self.l1(x)))


def loss_fn(model, batch):
    x, y = batch
    logits = model(x)
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, y)), None


def run_experiment(name, optimizer, devices, loss_fn, model, batch, steps, weights=None, calibrate=False):
    """Run one experiment configuration and report timing."""
    print(f"\n{'=' * 70}")
    print(f"  {name}")
    print(f"{'=' * 70}")

    with DistributedZeroGrad(optimizer, devices, loss_fn, weights=weights) as dist_opt:
        if calibrate:
            print("  Calibrating devices...")
            results = dist_opt.calibrate(model, batch, warmup=2, trials=3)
            for r in results:
                print(f"    {r.name}: {r.per_candidate_seconds*1000:.2f}ms/candidate")
            print(f"    → weights={[f'{w:.1f}' for w in dist_opt.weights]}")
            print(f"    → partition={dist_opt.partition_sizes}")
        else:
            print(f"  weights={dist_opt.weights}")
            print(f"  partition={dist_opt.partition_sizes}")

        state = dist_opt.init(model)
        t0 = time.time()
        for step in range(steps):
            model, state, metrics = dist_opt.step(state, model, batch)
            if step % 20 == 0 or step == steps - 1:
                print(f"    gen {metrics.generation:3d}  loss={metrics.mean_loss:.4f}  "
                      f"({(time.time() - t0) / (step + 1):.3f}s/step)")
        total = time.time() - t0
        print(f"    Total: {total:.1f}s ({total/steps:.3f}s/step)")
    return total, model


def main():
    parser = argparse.ArgumentParser(description="Asymmetric distributed ZeroGrad")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--pop", type=int, default=64)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--batch", type=int, default=256)
    args = parser.parse_args()

    cpu = jax.devices("cpu")[0]
    gpus = jax.devices("gpu")
    if gpus:
        gpu = gpus[0]
    else:
        print("Warning: no GPU found; falling back to CPU. The asymmetric "
              "compute demo is most meaningful with a GPU, but the protocol "
              "is device-agnostic.")
        gpu = cpu
    print(f"Devices: CPU={cpu.platform}:{cpu.id}, GPU={gpu.platform}:{gpu.id}")
    print(f"Model: {INPUT_DIM}→{HIDDEN_DIM}→{OUTPUT_DIM}, pop={args.pop}, batch={args.batch}")
    print(f"Steps: {args.steps}")

    BigMLP(nnx.Rngs(0))
    batch = (
        jax.random.normal(jax.random.key(1), (args.batch, INPUT_DIM)),
        jax.random.randint(jax.random.key(2), (args.batch,), 0, OUTPUT_DIM),
    )

    def make_optimizer():
        return ZeroGrad(
            optax.adamw(learning_rate=args.lr, weight_decay=0.0),
            population_size=args.pop,
            rank=args.rank,
            sigma=args.sigma,
            seed=42,
            run_id="asymmetric",
        )

    # Fresh models per experiment so init/surgery stays independent.
    _, _ = run_experiment(
        "1. Even split (naive)",
        make_optimizer(), [cpu, gpu], loss_fn,
        BigMLP(nnx.Rngs(0)), batch, args.steps,
    )

    _, _ = run_experiment(
        "2. Manual weights [1, 4] — GPU gets 4× more candidates",
        make_optimizer(), [cpu, gpu], loss_fn,
        BigMLP(nnx.Rngs(0)), batch, args.steps,
        weights=[1, 4],
    )

    _, _ = run_experiment(
        "3. Auto-calibrated weights",
        make_optimizer(), [cpu, gpu], loss_fn,
        BigMLP(nnx.Rngs(0)), batch, args.steps,
        calibrate=True,
    )

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print("""
For asymmetric compute (slow CPU + fast GPU, or mixed GPU generations):

  • Even split wastes the fast device — it idles waiting for the slow one.
  • Manual weights let you assign more candidates to faster devices.
  • Auto-calibration times each device and sets weights automatically.

For a 4× slow GPU + 1× fast GPU node:

    dist_opt = DistributedZeroGrad(
        opt,
        devices=[gpu0, gpu1, gpu2, gpu3, gpu4],
        loss_fn=loss_fn,
        weights=[1, 1, 1, 1, 4],  # fast GPU gets 4× the candidates
    )

Or let calibration figure it out:

    dist_opt.calibrate(model, batch)
""")


if __name__ == "__main__":
    main()
