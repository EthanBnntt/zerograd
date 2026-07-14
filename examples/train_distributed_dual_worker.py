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
import optax

from zerograd import DistributedZeroGrad, ZeroGrad
from _xor_model import XOR_X, XOR_Y, accuracy, build_manifest, build_params, loss_fn


def main():
    parser = argparse.ArgumentParser(description="Distributed ZeroGrad: dual GPU workers on XOR")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--pop", type=int, default=32)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--sigma", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=1e-2)
    args = parser.parse_args()

    gpus = jax.devices("gpu")
    if gpus:
        gpu = gpus[0]
        gpu_label = f"{gpu.platform} (id={gpu.id})"
    else:
        # No GPU available: fall back to CPU with a warning, instead of
        # crashing with an unhelpful IndexError (see issue #28).
        print("Warning: no GPU found; falling back to CPU for both workers.")
        gpu = jax.devices("cpu")[0]
        gpu_label = f"{gpu.platform} (CPU fallback)"
    # Two workers on the same device
    devices = [gpu, gpu]
    print(f"Worker 0: {gpu_label}")
    print(f"Worker 1: {gpu_label}  [shared]")
    print(f"Population: {args.pop} ({args.pop // 2} per worker)")
    print(f"Steps: {args.steps}\n")

    params = build_params(jax.random.key(0))
    manifest = build_manifest()
    batch = (XOR_X, XOR_Y)

    base_opt = ZeroGrad(
        manifest,
        optax.adamw(learning_rate=args.lr, weight_decay=0.0),
        population_size=args.pop,
        rank=args.rank,
        sigma=args.sigma,
        seed=42,
        run_id="xor-dual-gpu",
    )

    # Use the coordinator as a context manager so its ThreadPoolExecutor is
    # always shut down, even if the run is interrupted (see issue #27).
    with DistributedZeroGrad(
        base_opt, devices=devices, loss_fn=loss_fn,
    ) as dist_opt:
        state = dist_opt.init(params)

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
