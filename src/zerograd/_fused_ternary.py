"""Fused packed-ternary Linear+LUT forward for NVIDIA GPUs (Pallas/Triton).

Pure-integer ternary block: int8 activations × ternary ``{-1,0,+1}`` weights,
packed 4-per-byte into ``uint8`` and unpacked **on the fly** inside the kernel::

    int8 @ unpack(codes) -> int32 MMA accum   (Tensor Core; codes = W+1 ∈ {0,1,2})
    -> subtract rowsum(x)                     (codes→{-1,0,+1}; see note)
    -> floor-div by round(sqrt(K))            (ternary requantize, in-register)
    -> clip to [-127, 127] -> int8            (in-register)
    -> table[req + 128]                       (256-byte LUT, L1/SRAM resident)
    -> int8 store                             (single HBM write)

Algebra note (bit-identity): with ``codes = W + 1``,

    Σ_k x·codes = Σ_k x·(W+1) = Σ_k x·W + Σ_k x
    ⇒ Σ_k x·W = (x @ codes) − rowsum(x).

``rowsum(x)`` is weight-independent, so the kernel accumulates the non-negative
``x @ codes`` MMA and subtracts one row-sum vector — no per-element ``-1``
broadcast, and ``codes ∈ {0,1,2}`` never changes sign inside the MMA.

This mirrors :mod:`zerograd._fused_lut`, but the weight slab is the *packed*
``(K//4, N)`` uint8 array (4× less HBM traffic than an int8 weight) and each
``block_k`` tile is unpacked from ``block_k//4`` bytes in-register before the
``block_k × block_n`` dot.

Constraints (validated eagerly, fail loud), same as :mod:`zerograd._fused_lut`:
  * ``K`` must be a power of two and divisible by ``block_k``.
  * ``block_k`` must be divisible by ``TERNARY_PER_BYTE`` (4).
  * ``N`` must be a multiple of ``block_n``; ``M`` is padded to ``block_m``.

Dispatch (``mode="auto"``): the fused Triton kernel is CUDA-only (Pallas
Triton lowering). On non-GPU / non-Triton backends it falls back to the
bit-identical unfused reference. Tune ``_FUSED_MAX_K`` on the target GPU.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from ._integer import (
    TERNARY_PER_BYTE,
    int_matmul,
    ternary_matmul_divisor,
    ternary_requantize,
    unpack_ternary,
)

__all__ = [
    "fused_ternary_lut",
    "packed_ternary_lut_fused",
    "ternary_lut_reference",
]

_EGG_MIN, _EGG_MAX = -127, 127

#: Auto dispatch uses the fused kernel only at or below this K (same regime as
#: :mod:`zerograd._fused_lut`; re-measure the crossover on the target GPU).
_FUSED_MAX_K = 256


def ternary_lut_reference(
    x: jax.Array,
    w_packed: jax.Array,
    table: jax.Array,
    *,
    bias: jax.Array | None = None,
) -> jax.Array:
    """Unfused ground truth — bit-identical to ``TernaryLinearLUT.__call__``.

    ``x`` (..., K) int8, ``w_packed`` (K//4, N) uint8 packed ternary,
    ``table`` (256,) int8, optional ``bias`` (N,) int32. Returns (..., N) int8.
    """
    w = unpack_ternary(w_packed)
    acc = int_matmul(x, w)
    if bias is not None:
        acc = acc + bias
    req = ternary_requantize(acc, x.shape[-1], act_dtype=jnp.int8)
    idx = req.astype(jnp.int32) - jnp.iinfo(jnp.int8).min
    return table[idx]


def _validate_operands(
    x: jax.Array, w_packed: jax.Array, table: jax.Array, bias: jax.Array | None
) -> tuple[int, int, int]:
    if x.dtype != jnp.int8:
        raise TypeError(f"x must be int8, got {x.dtype}")
    if w_packed.dtype != jnp.uint8 or w_packed.ndim != 2:
        raise TypeError(
            f"w_packed must be a uint8 (K//4, N) array, got {w_packed.dtype} {w_packed.shape}"
        )
    if table.dtype != jnp.int8 or table.shape != (256,):
        raise ValueError(f"table must be int8 of shape (256,), got {table.dtype} {table.shape}")
    k4, n = w_packed.shape
    k = k4 * TERNARY_PER_BYTE
    if x.shape[-1] != k:
        raise ValueError(
            f"x inner dim {x.shape[-1]} != unpacked K {k} (packed rows {k4} × {TERNARY_PER_BYTE})"
        )
    if k < 1 or (k & (k - 1)) != 0:
        raise ValueError(f"in_features K={k} must be a power of two (Pallas block constraint)")
    if bias is not None and (bias.dtype != jnp.int32 or bias.shape != (n,)):
        raise ValueError(f"bias must be int32 of shape ({n},), got {bias.dtype} {bias.shape}")
    m = 1
    for d in x.shape[:-1]:
        m *= d
    return m, k, n


def _make_ternary_kernel(divisor: int, block_k: int, k_iters: int):
    """Trace a fused packed-ternary GEMM + requant + LUT kernel.

    Block refs: x (BM, K) row slab, w_packed (K//4, BN) packed column slab,
    bias (BN,), lut (256,) whole table, out (BM, BN). The K loop is a static
    Python loop over slices so the accumulator stays in registers — no int32
    partial sums ever touch HBM.
    """
    bk4 = block_k // TERNARY_PER_BYTE

    def kernel(x_ref, w_ref, bias_ref, lut_ref, o_ref):
        block_m = x_ref.shape[0]
        block_n = w_ref.shape[1]
        # (1, 4, 1) shift vector so ``ps[:, None, :] >> shifts`` broadcasts a
        # (bk4, BN) packed slab into a (bk4, 4, BN) codes plane without
        # ``stack`` (Pallas-Triton only supports 2-argument stack). Built
        # inside the kernel: captured constant arrays are not allowed. The
        # reshape (bk4, 4, BN) -> (block_k, BN) flattens q,i,n -> row 4q+i,
        # matching the byte packing order.
        shifts = (2 * jnp.arange(TERNARY_PER_BYTE, dtype=jnp.int32)).reshape(
            1, TERNARY_PER_BYTE, 1
        )
        acc = jnp.zeros((block_m, block_n), dtype=jnp.int32)
        rowsum = jnp.zeros((block_m,), dtype=jnp.int32)
        for kk in range(k_iters):
            xk = x_ref[:, kk * block_k : (kk + 1) * block_k]
            rowsum = rowsum + xk.astype(jnp.int32).sum(axis=1)
            ps = w_ref[kk * bk4 : (kk + 1) * bk4, :].astype(jnp.int32)
            codes = ((ps[:, None, :] >> shifts) & 3).reshape(block_k, block_n)
            acc = acc + jax.lax.dot_general(
                xk,
                codes.astype(jnp.int8),
                (((1,), (0,)), ((), ())),
                preferred_element_type=jnp.int32,
            )
        acc = acc - rowsum[:, None]
        acc = acc + bias_ref[...][None, :]
        scaled = jnp.floor_divide(acc, divisor)
        req = jnp.clip(scaled, _EGG_MIN, _EGG_MAX).astype(jnp.int8)
        idx = req.astype(jnp.int32) - jnp.iinfo(jnp.int8).min
        o_ref[...] = lut_ref[idx]

    return kernel


def fused_ternary_lut(
    x: jax.Array,
    w_packed: jax.Array,
    table: jax.Array,
    *,
    bias: jax.Array | None = None,
    mode: str = "auto",
    block_m: int = 256,
    block_n: int = 64,
    block_k: int = 256,
    num_warps: int = 8,
    num_stages: int = 2,
    allow_fallback: bool = True,
) -> jax.Array:
    """Fused int8 × packed-ternary linear + EGG requantize + int8 LUT (GPU only).

    Args:
        x: (..., K) int8 activations.
        w_packed: (K//4, N) uint8 packed ternary weights (see
            :func:`zerograd.pack_ternary`).
        table: (256,) int8 LUT (indexed by ``requantized + 128``).
        bias: optional (N,) int32 added to the accumulator before requantize.
        mode: "auto" (fused kernel where it wins: GPU + K <= _FUSED_MAX_K),
            "fused" (force the Pallas kernel), "reference" (force unfused XLA).
        block_m/block_n/block_k: kernel tile sizes (powers of two;
            ``block_k`` must divide ``K`` and be divisible by 4).
        num_warps/num_stages: Triton launch parameters.
        allow_fallback: when ``mode="fused"`` but no GPU/Triton is present,
            fall back to the unfused reference instead of raising.

    Returns:
        (..., N) int8, bit-identical to :func:`ternary_lut_reference`.
    """
    m, k, n = _validate_operands(x, w_packed, table, bias)
    if mode not in ("auto", "fused", "reference"):
        raise ValueError(f"mode must be auto|fused|reference, got {mode!r}")
    # Tile validation is mode-independent so every path fails identically.
    block_n = min(block_n, n)
    block_k = min(block_k, k)
    if block_k % TERNARY_PER_BYTE != 0:
        block_k = max(TERNARY_PER_BYTE, block_k - (block_k % TERNARY_PER_BYTE))
    if n % block_n != 0:
        raise ValueError(f"out_features N={n} must be a multiple of block_n={block_n}")
    if k % block_k != 0:
        raise ValueError(f"K={k} must be a multiple of block_k={block_k}")
    if block_k % TERNARY_PER_BYTE != 0:
        raise ValueError(f"block_k={block_k} must be divisible by {TERNARY_PER_BYTE}")

    if mode == "reference":
        return ternary_lut_reference(x, w_packed, table, bias=bias)
    on_gpu = jax.default_backend() == "gpu"
    use_fused = mode == "fused" or (on_gpu and k <= _FUSED_MAX_K)
    if not (on_gpu and use_fused):
        if mode == "fused" and not allow_fallback:
            raise RuntimeError("fused_ternary_lut requires a GPU (Pallas/Triton)")
        return ternary_lut_reference(x, w_packed, table, bias=bias)

    # Deferred: jax.experimental.pallas.triton requires the triton package,
    # which CPU/ROCm users do not have installed.
    try:
        from jax.experimental.pallas import triton as pltr
    except Exception:  # pragma: no cover - backend dependent
        if mode == "fused" and not allow_fallback:
            raise RuntimeError("fused_ternary_lut requires jax.experimental.pallas.triton")
        return ternary_lut_reference(x, w_packed, table, bias=bias)

    x2 = x.reshape(m, k)
    pad_m = (-m) % block_m
    if pad_m:
        x2 = jnp.pad(x2, ((0, pad_m), (0, 0)))
    m_pad = m + pad_m
    bias_arg = bias if bias is not None else jnp.zeros((n,), dtype=jnp.int32)

    kernel = _make_ternary_kernel(ternary_matmul_divisor(k), block_k, k // block_k)
    fn = pl.pallas_call(
        kernel,
        grid=(m_pad // block_m, n // block_n),
        in_specs=[
            pl.BlockSpec((block_m, k), lambda i, j: (i, 0)),
            pl.BlockSpec((k // TERNARY_PER_BYTE, block_n), lambda i, j: (0, j)),
            pl.BlockSpec((block_n,), lambda i, j: (j,)),
            pl.BlockSpec((256,), lambda i, j: (0,)),
        ],
        out_specs=pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
        out_shape=jax.ShapeDtypeStruct((m_pad, n), jnp.int8),
        compiler_params=pltr.CompilerParams(num_warps=num_warps, num_stages=num_stages),
    )
    out = fn(x2, w_packed, bias_arg, table)
    if pad_m:
        out = out[:m]
    return out.reshape(*x.shape[:-1], n)


def packed_ternary_lut_fused(module, x: jax.Array) -> jax.Array:
    """Apply a ``TernaryLinearLUT`` module via the fused packed-ternary kernel.

    Packs ``module.linear.kernel`` (int8 ``{-1,0,1}``) to uint8 and runs the
    fused kernel with ``module.linear.bias`` / ``module.lut.table``. The
    unfused module and this function are interchangeable in the forward pass
    (bit-identical outputs); use it for eval/inference fast paths.
    """
    from ._integer import pack_ternary

    linear = module.linear
    kernel = linear.kernel[...]
    bias = linear.bias[...] if (linear.use_bias and linear.bias is not None) else None
    lut = module.lut
    table = lut.table[...]
    return fused_ternary_lut(x, pack_ternary(kernel), table, bias=bias)
