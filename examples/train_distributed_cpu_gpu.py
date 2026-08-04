"""Distributed ZeroGrad: CPU + GPU workers on XOR.

Splits the ES population across two workers:
  - Worker 0: CPU device (CpuDevice)
  - Worker 1: GPU device (RocmDevice / CudaDevice)

Each worker evaluates its half of the candidates independently.  Only the
1D loss arrays (a handful of floats) cross the device boundary.  The
coordinator gathers losses, shapes them, and completes the optimizer step
on the GPU.

    uv run python examples/train_distributed_cpu_gpu.py [--steps N]

This is a thin, fixed-topology wrapper around
``train_distributed_xor.py --devices cpu,gpu`` (see also
``train_distributed_dual_worker.py``).
"""

from __future__ import annotations

from train_distributed_xor import main

if __name__ == "__main__":
    main(default_devices="cpu,gpu", default_run_id="xor-cpu-gpu")
