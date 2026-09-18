"""Distributed ZeroGrad on XOR — configurable device topology.

Splits the ES population across N workers, each pinned to a device given by
``--devices`` (e.g. ``cpu,gpu`` for one CPU + one GPU worker, or ``gpu,gpu``
to simulate two workers sharing a single GPU — the setup you'd use as a
stand-in for a real multi-GPU node, see ``--weights`` for asymmetric splits).

Each worker evaluates its shard of the candidates independently. Only the
1D loss arrays (a handful of floats) cross the device boundary. The
coordinator gathers losses, shapes them, and completes the optimizer step
on the first device.

This demonstrates the core distributed property of ES optimization:
candidates are embarrassingly parallel, and workers need to share only
fitness values — not parameters, gradients, or activations.

    uv run python examples/train_distributed_xor.py --layout cpu_gpu
    uv run python examples/train_distributed_xor.py --layout dual_gpu
    uv run python examples/train_distributed_xor.py --devices cpu,gpu --weights 1,4
"""

from __future__ import annotations

import argparse
import time

import jax
import optax
from _xor_model import XOR_X, XOR_Y, accuracy, build_model, loss_fn

from zerograd import DistributedZeroGrad, ZeroGrad


def _parse_device_spec(spec: str) -> jax.Device:
    """Parse one ``--devices`` token, e.g. ``"cpu"``, ``"gpu"``, ``"gpu:1"``."""
    spec = spec.strip().lower()
    kind, _, idx_str = spec.partition(":")
    if kind not in ("cpu", "gpu"):
        raise SystemExit(f"unknown device kind {kind!r} in {spec!r} (expected 'cpu' or 'gpu')")
    idx = int(idx_str) if idx_str else 0

    pool = jax.devices(kind)
    if kind == "gpu" and not pool:
        # No GPU available: fall back to CPU with a warning, instead of
        # crashing with an unhelpful IndexError (see issue #28).
        print(f"Warning: no GPU found for {spec!r}; falling back to CPU.")
        kind, pool = "cpu", jax.devices("cpu")
        idx = min(idx, len(pool) - 1)
    if idx >= len(pool):
        raise SystemExit(
            f"requested {spec!r} but only {len(pool)} {kind} device(s) are visible"
        )
    return pool[idx]


def resolve_devices(specs: list[str]) -> list[jax.Device]:
    """Resolve ``--devices`` (e.g. ``["cpu", "gpu"]``) to JAX devices."""
    if len(specs) < 2:
        raise SystemExit("--devices needs at least two entries (one per worker)")
    return [_parse_device_spec(spec) for spec in specs]


def print_topology(devices: list[jax.Device]) -> None:
    for i, device in enumerate(devices):
        shared = "  [shared]" if devices[:i].count(device) > 0 else ""
        print(f"Worker {i}: {device.platform} (id={device.id}){shared}")


def run(
    *,
    devices: list[jax.Device],
    weights: list[float] | None,
    steps: int,
    pop: int,
    rank: int,
    sigma: float,
    lr: float,
    run_id: str,
) -> None:
    print(f"Population: {pop} across {len(devices)} workers "
          f"(weights={weights!r})")
    print(f"Steps: {steps}\n")
    print_topology(devices)
    print()

    model = build_model(jax.random.key(0))
    batch = (XOR_X, XOR_Y)

    base_opt = ZeroGrad(
        optax.adamw(learning_rate=lr, weight_decay=0.0),
        population_size=pop,
        rank=rank,
        sigma=sigma,
        seed=42,
        run_id=run_id,
    )

    # Use the coordinator as a context manager so its ThreadPoolExecutor is
    # always shut down, even if the run is interrupted (see issue #27).
    with DistributedZeroGrad(
        base_opt, devices=devices, loss_fn=loss_fn, weights=weights,
    ) as dist_opt:
        state = dist_opt.init(model)

        for shard in dist_opt.shards:
            print(f"  {shard.name}: {shard.device}")

        print()
        t0 = time.time()
        for step in range(steps):
            model, state, metrics = dist_opt.step(state, model, batch)
            if step % 50 == 0 or step == steps - 1:
                acc = accuracy(model)
                print(f"  gen {metrics.generation:3d}  "
                      f"loss={metrics.mean_loss:.4f}  "
                      f"acc={acc:.0%}  "
                      f"({(time.time() - t0) / (step + 1):.2f}s/step)")

        print(f"\nFinal accuracy: {accuracy(model):.0%}")
        print(f"Total time: {time.time() - t0:.1f}s")


_LAYOUTS = {
    "cpu_gpu": ("cpu,gpu", "xor-cpu-gpu"),
    "dual_gpu": ("gpu,gpu", "xor-dual-gpu"),
}


def build_arg_parser(
    *,
    default_devices: str | None = None,
    default_run_id: str = "xor-distributed",
    default_layout: str | None = "cpu_gpu",
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Distributed ZeroGrad on XOR")
    parser.add_argument(
        "--layout",
        choices=sorted(_LAYOUTS),
        default=default_layout,
        help="Preset worker topology (cpu_gpu | dual_gpu). "
        "Ignored when --devices is set explicitly.",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default=default_devices,
        help="Comma-separated device specs, one per worker "
        "(e.g. 'cpu,gpu' or 'gpu,gpu' or 'gpu:0,gpu:1'). "
        "Overrides --layout when provided.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Optional comma-separated per-worker weights for asymmetric "
        "population splits (e.g. '1,4' gives the second worker 4x candidates).",
    )
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--pop", type=int, default=32)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--sigma", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--run-id", type=str, default=default_run_id)
    return parser


def main(
    *,
    default_devices: str | None = None,
    default_run_id: str | None = None,
    default_layout: str | None = "cpu_gpu",
) -> None:
    parser = build_arg_parser(
        default_devices=default_devices,
        default_run_id=default_run_id or "xor-distributed",
        default_layout=default_layout,
    )
    args = parser.parse_args()

    layout = args.layout or "cpu_gpu"
    layout_devices, layout_run_id = _LAYOUTS[layout]
    devices_str = args.devices or layout_devices
    run_id = args.run_id
    if not args.devices and run_id == "xor-distributed":
        run_id = layout_run_id

    device_specs = [spec for spec in devices_str.split(",") if spec.strip()]
    devices = resolve_devices(device_specs)
    weights = (
        [float(w) for w in args.weights.split(",")] if args.weights else None
    )
    if weights is not None and len(weights) != len(devices):
        raise SystemExit(
            f"--weights has {len(weights)} entries but --devices has {len(devices)}"
        )

    run(
        devices=devices,
        weights=weights,
        steps=args.steps,
        pop=args.pop,
        rank=args.rank,
        sigma=args.sigma,
        lr=args.lr,
        run_id=run_id,
    )


if __name__ == "__main__":
    main()
