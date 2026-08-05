"""Distributed ZeroGrad: two GPU workers on XOR.

Runs two workers on the same GPU, each evaluating half the population.
This demonstrates the protocol for a multi-GPU node where each GPU runs
one worker — here we simulate it with two shards on one GPU.

The key property: each worker evaluates its candidates independently and
shares only its 1D loss array.  Whether the workers are on separate GPUs
or the same GPU, the protocol is identical.

For a real multi-GPU setup (e.g. 4× GPU node), pass one device per GPU to
``train_distributed_xor.py --devices gpu:0,gpu:1,gpu:2,gpu:3`` — no other
changes needed.

    uv run python examples/train_distributed_dual_worker.py [--steps N]

This is a thin, fixed-topology wrapper around
``train_distributed_xor.py --devices gpu,gpu`` (see also
``train_distributed_cpu_gpu.py``).
"""

from __future__ import annotations

from train_distributed_xor import main

if __name__ == "__main__":
    main(default_devices="gpu,gpu", default_run_id="xor-dual-gpu")
