"""Distributed ZeroGrad: CPU + GPU workers on XOR.

Splits the ES population across two workers:
  - Worker 0: CPU device (CpuDevice)
  - Worker 1: GPU device (RocmDevice / CudaDevice)

Each worker evaluates its half of the candidates independently.  Only the
1D loss arrays (a handful of floats) cross the device boundary.  The
coordinator gathers losses, shapes them, and completes the optimizer step
on the GPU.

This demonstrates the core distributed property of ES optimization:
candidates are embarrassingly parallel, and workers need to share only
fitness values — not parameters, gradients, or activations.

    uv run python examples/train_distributed_cpu_gpu.py [--steps N]
"""

from __future__ import annotations

import argparse
import time

import jax
import optax

from zerograd import DistributedZeroGrad, ZeroGrad
from _xor_model import XOR_X, XOR_Y, accuracy, build_manifest, build_params, loss_fn


def main():
    parser = argparse.ArgumentParser(description="Distributed ZeroGrad: CPU + GPU on XOR")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--pop", type=int, default=32)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--sigma", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=1e-2)
    args = parser.parse_args()

    cpu = jax.devices("cpu")[0]
    gpus = jax.devices("gpu")
    if gpus:
        gpu = gpus[0]
    else:
        # No GPU available: fall back to CPU for both workers with a warning,
        # instead of crashing with an unhelpful IndexError (see issue #28).
        print("Warning: no GPU found; falling back to CPU for both workers.")
        gpu = cpu
    print(f"Worker 0: {cpu.platform} (id={cpu.id})")
    print(f"Worker 1: {gpu.platform} (id={gpu.id})")
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
        run_id="xor-cpu-gpu",
    )

    # Use the coordinator as a context manager so its ThreadPoolExecutor is
    # always shut down, even if the run is interrupted (see issue #27).
    with DistributedZeroGrad(
        base_opt, devices=[cpu, gpu], loss_fn=loss_fn,
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
