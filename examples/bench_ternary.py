"""Benchmark: fastest packed-ternary linear layer (int8 @ ternary -> int8 LUT).

Two separate questions, measured over a grid of (M, K, N):

  A. Raw matmul (int8 @ ternary -> int32). Baseline ``dense`` (cuBLAS int8 MMA):
       dense      — unpack to int8, ``int_matmul``.
       packed_xla — unpack uint8->int8 in XLA bit-ops, then int_matmul.
       sign_split — x@(W==1) - x@(W==-1) (two binary-masked matmuls).
       sparse     — jax.experimental.sparse BCOO (sparsity-exploiting).

  B. Full layer (matmul + ternary requantize + LUT). This is where the fused
     kernel pays off: it removes the int32 accumulator's HBM roundtrip.
       unfused    — int_matmul -> ternary_requantize -> LUT (XLA, roundtrips).
       fused      — single Pallas/Triton kernel, packed uint8 unpacked on the
                    fly, single HBM write (GPU only).

Every method is checked bit-identical to its reference before timing. The
``fused`` speedup (section B) is reported vs the *unfused full layer*.

    uv run python examples/bench_ternary.py            # standard grid
    uv run python examples/bench_ternary.py --big      # add large shapes
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

from zerograd import (
    fused_ternary_lut,
    pack_ternary,
    unpack_ternary,
)
from zerograd._integer import int_matmul, ternary_requantize

_IDENTITY_LUT = jnp.arange(-128, 128, dtype=jnp.int8)


# ── A. raw matmul variants ───────────────────────────────────────────────────
def mm_dense(x, w):
    return int_matmul(x, w)


def mm_packed_xla(x, packed):
    return int_matmul(x, unpack_ternary(packed))


def mm_sign_split(x, w):
    pos = (w == 1).astype(jnp.int8)
    neg = (w == -1).astype(jnp.int8)
    return int_matmul(x, pos) - int_matmul(x, neg)


def mm_sparse_bcoo(x, w_sp):
    return (x.astype(jnp.int32) @ w_sp).astype(jnp.int32)


# ── B. full layer variants ───────────────────────────────────────────────────
def layer_unfused(x, w, table):
    req = ternary_requantize(int_matmul(x, w), x.shape[-1], act_dtype=jnp.int8)
    return table[req.astype(jnp.int32) + 128]


def layer_fused(x, packed, table, kw):
    return fused_ternary_lut(x, packed, table, mode="fused", allow_fallback=False, **kw)


def _bench(fn, args, iters, warmup):
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    t0 = time.perf_counter()
    for _ in range(iters):
        out = jax.block_until_ready(fn(*args))
    del out
    return (time.perf_counter() - t0) / iters


def _tops(m, k, n, secs):
    return (2.0 * m * k * n) / secs / 1e12


def _cell(m, k, n, secs, base):
    if secs is None or np.isnan(secs):
        return f"{'n/a':>15}"
    return f"{_tops(m,k,n,secs):6.2f}T {base/secs:5.2f}x"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--big", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"backend={jax.default_backend()}  devices={jax.devices()}", flush=True)
    on_gpu = jax.default_backend() == "gpu"

    shapes = [
        (256, 256, 256),
        (1024, 256, 256),
        (4096, 256, 256),
        (8192, 256, 256),
        (1024, 512, 512),
        (2048, 1024, 1024),
    ]
    if args.big:
        shapes += [(16384, 256, 256), (4096, 4096, 1024), (8192, 2048, 2048)]

    # Fused tile configs to sweep (block_m, block_n, block_k, warps, stages).
    tile_cfgs = [
        dict(block_m=128, block_n=64, block_k=256, num_warps=8, num_stages=2),
        dict(block_m=256, block_n=64, block_k=256, num_warps=8, num_stages=2),
        dict(block_m=128, block_n=128, block_k=128, num_warps=8, num_stages=3),
        dict(block_m=64, block_n=64, block_k=256, num_warps=4, num_stages=2),
    ]

    for (m, k, n) in shapes:
        key = jax.random.PRNGKey(args.seed + m + k + n)
        kx, kw = jax.random.split(key)
        x = jax.random.randint(kx, (m, k), -128, 128, jnp.int8)
        w = jax.random.randint(kw, (k, n), -1, 2, jnp.int8)
        packed = pack_ternary(w)
        table = _IDENTITY_LUT

        ref_acc = mm_dense(x, w)
        ref_layer = layer_unfused(x, w, table)

        # ── A: raw matmul ────────────────────────────────────────────────
        a = {}
        a["dense"] = _bench(jax.jit(mm_dense), (x, w), args.iters, args.warmup)
        got = mm_packed_xla(x, packed)
        assert bool(jnp.all(got == ref_acc)), "packed_xla mismatch"
        a["packed_xla"] = _bench(jax.jit(mm_packed_xla), (x, packed), args.iters, args.warmup)
        got = mm_sign_split(x, w)
        assert bool(jnp.all(got == ref_acc)), "sign_split mismatch"
        a["sign_split"] = _bench(jax.jit(mm_sign_split), (x, w), args.iters, args.warmup)
        nnz = float("nan")
        if k * n <= 1 << 20 and m <= 4096:  # sparse only worth/tractable when small
            try:
                from jax.experimental import sparse

                w_sp = sparse.BCOO.fromdense(w.astype(jnp.int8))
                nnz = float(w_sp.nse) / (k * n)
                assert bool(jnp.all(mm_sparse_bcoo(x, w_sp) == ref_acc)), "sparse mismatch"
                a["sparse"] = _bench(jax.jit(mm_sparse_bcoo), (x, w_sp), args.iters, args.warmup)
            except Exception as e:  # noqa: BLE001
                a["sparse"] = float("nan")
                print(f"  sparse failed: {type(e).__name__}", flush=True)
        else:
            a["sparse"] = float("nan")
        base_a = a["dense"]
        row_a = " ".join(
            _cell(m, k, n, a[name], base_a)
            for name in ["dense", "packed_xla", "sign_split", "sparse"]
        )

        # ── B: full layer (unfused vs fused) ─────────────────────────────
        b_unfused = _bench(jax.jit(layer_unfused), (x, w, table), args.iters, args.warmup)
        best_fused = float("nan")
        best_cfg = None
        if on_gpu:
            for cfg in tile_cfgs:
                if k % cfg["block_k"] != 0 or n % cfg["block_n"] != 0:
                    continue
                try:
                    fl = layer_fused(x, packed, table, cfg)
                    if not bool(jnp.all(fl == ref_layer)):
                        print(f"  fused mismatch cfg={cfg}", flush=True)
                        continue
                    t = _bench(
                        jax.jit(lambda a, b: layer_fused(a, b, table, cfg)),
                        (x, packed),
                        args.iters,
                        args.warmup,
                    )
                    if np.isnan(best_fused) or t < best_fused:
                        best_fused, best_cfg = t, cfg
                except Exception as e:  # noqa: BLE001
                    print(f"  fused cfg={cfg} failed: {type(e).__name__}", flush=True)
        spd = (b_unfused / best_fused) if not np.isnan(best_fused) else float("nan")
        print(
            f"{m:>6} {k:>6} {n:>6} | A: {row_a} | "
            f"B: unfused {_tops(m,k,n,b_unfused):6.2f}T  fused {_tops(m,k,n,best_fused):6.2f}T  "
            f"speedup {spd:4.2f}x  nnz={nnz:.0%}",
            flush=True,
        )
        if best_cfg:
            print(f"         best fused cfg: {best_cfg}", flush=True)


if __name__ == "__main__":
    main()
