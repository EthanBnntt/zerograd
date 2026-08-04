"""Microbench: GDN-2 chunkwise VRAM + fused int8 in-proj shapes on the active device.

Run on H100 after ``scripts/setup_h100_cuda.sh``::

    .venv/bin/python examples/bench_gdn_int_gemm.py
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

_ROOT = Path(__file__).resolve().parents[1]
_TRAIN = _ROOT / "examples" / "train_int_rnn_minipile.py"
_spec = importlib.util.spec_from_file_location("train_int_rnn_minipile", _TRAIN)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
sys.modules["train_int_rnn_minipile"] = _mod
_spec.loader.exec_module(_mod)


def _rand_q8(key, shape, *, gate: bool = False):
    # Keep gates away from 0 so long-horizon WY vs scan stays within ±1 int8.
    if gate:
        return jax.random.randint(key, shape, 64, 128, dtype=jnp.int32).astype(jnp.int8)
    return jax.random.randint(key, shape, -64, 64, dtype=jnp.int32).astype(jnp.int8)


def _peak_bytes() -> int:
    try:
        return int(jax.local_devices()[0].memory_stats()["peak_bytes_in_use"])
    except Exception:
        return -1


def bench_wy(bh: int, t: int, d: int, c: int, tile: int, reps: int = 5) -> None:
    key = jax.random.key(0)
    keys = jax.random.split(key, 6)
    q = _rand_q8(keys[0], (bh, t, d))
    k = _rand_q8(keys[1], (bh, t, d))
    v = _rand_q8(keys[2], (bh, t, d))
    a = _rand_q8(keys[3], (bh, t, d), gate=True)
    b = _rand_q8(keys[4], (bh, t, d), gate=True)
    w = _rand_q8(keys[5], (bh, t, d), gate=True)

    @jax.jit
    def run(q, k, v, a, b, w):
        return _mod._gdn2_chunkwise_wy(q, k, v, a, b, w, chunk_size=c, feat_tile=tile)

    y = run(q, k, v, a, b, w)
    jax.block_until_ready(y)
    peak0 = _peak_bytes()

    t0 = time.perf_counter()
    for _ in range(reps):
        y = run(q, k, v, a, b, w)
    jax.block_until_ready(y)
    dt = (time.perf_counter() - t0) / reps
    peak1 = _peak_bytes()

    ys = _mod._gdn2_stepwise_scan(q, k, v, a, b, w)
    diff = np.abs(np.asarray(y, np.int16) - np.asarray(ys, np.int16))
    print(
        f"WY bh={bh} T={t} d={d} C={c} tile={tile}: "
        f"{dt*1e3:.2f} ms/iter  peak_bytes≈{peak1} (Δ {peak1 - peak0})  "
        f"vs_stepwise max|Δ|={int(diff.max())} fracΔ={(diff > 0).mean():.4f}",
        flush=True,
    )


def bench_in_proj(batch: int, seq: int, dim: int, heads: int, reps: int = 5) -> None:
    _mod.configure_architecture(dim=dim, heads=heads, layers=1, ffn_mult=4, seq_len=seq)
    mixer = _mod.MultiHeadGatedDelta2Mixer(rngs=nnx.Rngs(0), impl="chunkwise")
    x = jax.random.randint(jax.random.key(1), (batch, seq, dim), -64, 64, dtype=jnp.int8)

    @jax.jit
    def run(x):
        return mixer(x)

    y = run(x)
    jax.block_until_ready(y)
    t0 = time.perf_counter()
    for _ in range(reps):
        y = run(x)
    jax.block_until_ready(y)
    dt = (time.perf_counter() - t0) / reps
    m = batch * seq
    print(
        f"mixer B={batch} T={seq} D={dim}: {dt*1e3:.2f} ms/iter  "
        f"in_proj IU8 GEMM ~ [{m}×{dim}]@[{dim}×{6*dim}]  "
        f"out={tuple(y.shape)} {y.dtype}",
        flush=True,
    )


def main() -> None:
    print(
        f"devices={jax.devices()} backend={jax.default_backend()} "
        f"feat_tile={_mod.GDN_FEAT_TILE}",
        flush=True,
    )
    bench_wy(bh=16 * 16, t=512, d=128, c=64, tile=32)
    bench_wy(bh=16 * 16, t=512, d=128, c=64, tile=16)
    bench_in_proj(batch=32, seq=512, dim=2048, heads=16)


if __name__ == "__main__":
    main()
