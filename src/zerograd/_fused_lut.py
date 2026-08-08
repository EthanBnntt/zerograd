"""Fused int8 Linear+LUT forward for NVIDIA GPUs (Pallas/Triton kernel).

Fuses the :class:`~zerograd.IntLinearLUT` forward into a single kernel so the
int32 GEMM accumulator never leaves the SM before the pointwise stage::

    int8 @ int8 -> int32 MMA accum        (Tensor Core)
    -> floor-div by round(16*sqrt(K))     (EGG requantize, in-register)
    -> clip to [-127, 127] -> int8        (in-register)
    -> table[req + 128]                   (256-byte LUT, L1/SRAM resident)
    -> int8 store                         (single HBM write)

The unfused XLA path materializes the int32 GEMM output in HBM, then runs
requantize+gather as one or more separate pointwise/gather kernels — two extra
global-memory roundtrips per layer. Fusion removes them, which matters for ES
evaluation loops where this block runs millions of times with no backprop.

Backend notes (jax >= 0.10): ``pallas_call`` defaults to the Mosaic GPU
lowering, which cannot express the per-element LUT gather. We therefore pin
the Triton backend via ``pltr.CompilerParams`` — it lowers ``int8 @ int8`` to
Hopper IMMA Tensor Core ops and ``lut_ref[idx]`` to a pointer-arithmetic
gather (the 256-byte table stays L1-resident across all blocks on the SM).

Constraints (validated eagerly, fail loud):
  * ``K`` (in_features) must be a power of two — the Pallas Triton lowering
    requires power-of-two-sized block arrays.
  * ``N`` (out_features) must be a multiple of ``block_n``.
  * ``M`` (rows) is padded internally to a multiple of ``block_m``.

Dispatch (``mode="auto"``): measured on H100, the fused Triton kernel wins
for ``K <= _FUSED_MAX_K`` — there the int32 GEMM roundtrip through HBM
dominates and epilogue fusion removes it (1.15-1.7x at large batch, up to
~200 TOPS). For larger K the Pallas-Triton int8 GEMM core cannot keep up
with cuBLAS (~1000 TOPS), so auto mode falls back to the unfused XLA path,
whose requantize+LUT tail XLA already fuses into one pointwise pass. Both
paths are bit-identical, so dispatch is a pure performance decision.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from ._integer import egg_matmul_divisor, egg_requantize, int_matmul

__all__ = ["fused_linear_lut", "int_linear_lut_fused", "linear_lut_reference"]

_EGG_MIN, _EGG_MAX = -127, 127

#: Auto dispatch uses the fused kernel only at or below this K (H100-measured
#: crossover with tuned tiles bm=256/bn=64/w8: fused wins 1.15-1.7x at
#: K <= 256 across N in {128..1536}, loses 0.7x at K=512; see module docstring).
_FUSED_MAX_K = 256


def linear_lut_reference(
    x: jax.Array,
    w: jax.Array,
    table: jax.Array,
    *,
    bias: jax.Array | None = None,
) -> jax.Array:
    """Unfused ground truth — bit-identical to ``IntLinearLUT.__call__``.

    ``x`` (..., K) int8, ``w`` (K, N) int8, ``table`` (256,) int8,
    optional ``bias`` (N,) int32. Returns (..., N) int8.
    """
    acc = int_matmul(x, w)
    if bias is not None:
        acc = acc + bias
    req = egg_requantize(acc, x.shape[-1], act_dtype=jnp.int8)
    idx = req.astype(jnp.int32) - jnp.iinfo(jnp.int8).min
    return table[idx]


def _validate_operands(
    x: jax.Array, w: jax.Array, table: jax.Array, bias: jax.Array | None
) -> tuple[int, int, int]:
    if x.dtype != jnp.int8 or w.dtype != jnp.int8:
        raise TypeError(f"x and w must be int8, got {x.dtype} and {w.dtype}")
    if table.dtype != jnp.int8 or table.shape != (256,):
        raise ValueError(f"table must be int8 of shape (256,), got {table.dtype} {table.shape}")
    k, n = w.shape
    if x.shape[-1] != k:
        raise ValueError(f"x inner dim {x.shape[-1]} != w rows {k}")
    if k < 1 or (k & (k - 1)) != 0:
        raise ValueError(f"in_features K={k} must be a power of two (Pallas block constraint)")
    if bias is not None and (bias.dtype != jnp.int32 or bias.shape != (n,)):
        raise ValueError(f"bias must be int32 of shape ({n},), got {bias.dtype} {bias.shape}")
    m = 1
    for d in x.shape[:-1]:
        m *= d
    return m, k, n


def _make_kernel(divisor: int, block_k: int, k_iters: int):
    """Trace a fused GEMM+requant+LUT kernel with static unrolled K loop.

    Block refs: x (BM, K) row slab, w (K, BN) column slab, bias (BN,),
    lut (256,) whole table, out (BM, BN). The K loop is a static Python
    loop over slices so the accumulator stays in registers — no int32
    partial sums ever touch HBM.
    """

    def kernel(x_ref, w_ref, bias_ref, lut_ref, o_ref):
        acc = jnp.zeros((x_ref.shape[0], w_ref.shape[1]), dtype=jnp.int32)
        for kk in range(k_iters):
            sl = slice(kk * block_k, (kk + 1) * block_k)
            acc = acc + jax.lax.dot_general(
                x_ref[:, sl],
                w_ref[sl, :],
                (((1,), (0,)), ((), ())),
                preferred_element_type=jnp.int32,
            )
        acc = acc + bias_ref[...][None, :]
        scaled = jnp.floor_divide(acc, divisor)
        req = jnp.clip(scaled, _EGG_MIN, _EGG_MAX).astype(jnp.int8)
        idx = req.astype(jnp.int32) - jnp.iinfo(jnp.int8).min
        o_ref[...] = lut_ref[idx]

    return kernel


def fused_linear_lut(
    x: jax.Array,
    w: jax.Array,
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
    """Fused int8 linear + EGG requantize + int8 LUT (GPU only).

    Args:
        x: (..., K) int8 activations.
        w: (K, N) int8 weights.
        table: (256,) int8 LUT (indexed by ``requantized + 128``).
        bias: optional (N,) int32 added to the accumulator before requantize.
        mode: "auto" (fused kernel where it wins: GPU + K <= _FUSED_MAX_K),
            "fused" (force the Pallas kernel), "reference" (force unfused XLA).
        block_m/block_n/block_k: kernel tile sizes (powers of two;
            ``block_k`` must divide ``K``). Defaults tuned on H100 for the
            small-K regime where the fused path dispatches.
        num_warps/num_stages: Triton launch parameters.
        allow_fallback: when ``mode="fused"`` but no GPU is present, fall
            back to the jitted unfused reference instead of raising.

    Returns:
        (..., N) int8, bit-identical to :func:`linear_lut_reference`.
    """
    m, k, n = _validate_operands(x, w, table, bias)
    if mode not in ("auto", "fused", "reference"):
        raise ValueError(f"mode must be auto|fused|reference, got {mode!r}")
    # Tile validation is mode-independent so every path fails identically.
    block_n = min(block_n, n)
    block_k = min(block_k, k)
    if n % block_n != 0:
        raise ValueError(f"out_features N={n} must be a multiple of block_n={block_n}")
    if k % block_k != 0:
        raise ValueError(f"K={k} must be a multiple of block_k={block_k}")

    if mode == "reference":
        return linear_lut_reference(x, w, table, bias=bias)
    on_gpu = jax.default_backend() == "gpu"
    use_fused = mode == "fused" or (on_gpu and k <= _FUSED_MAX_K)
    if not (on_gpu and use_fused):
        if mode == "fused" and not allow_fallback:
            raise RuntimeError("fused_linear_lut requires a GPU (Pallas/Triton)")
        return linear_lut_reference(x, w, table, bias=bias)

    # Deferred: jax.experimental.pallas.triton requires the triton package,
    # which CPU/ROCm users do not have installed.
    from jax.experimental.pallas import triton as pltr

    x2 = x.reshape(m, k)
    pad_m = (-m) % block_m
    if pad_m:
        x2 = jnp.pad(x2, ((0, pad_m), (0, 0)))
    m_pad = m + pad_m
    bias_arg = bias if bias is not None else jnp.zeros((n,), dtype=jnp.int32)

    kernel = _make_kernel(egg_matmul_divisor(k), block_k, k // block_k)
    fn = pl.pallas_call(
        kernel,
        grid=(m_pad // block_m, n // block_n),
        in_specs=[
            pl.BlockSpec((block_m, k), lambda i, j: (i, 0)),
            pl.BlockSpec((k, block_n), lambda i, j: (0, j)),
            pl.BlockSpec((block_n,), lambda i, j: (j,)),
            pl.BlockSpec((256,), lambda i, j: (0,)),
        ],
        out_specs=pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
        out_shape=jax.ShapeDtypeStruct((m_pad, n), jnp.int8),
        compiler_params=pltr.CompilerParams(num_warps=num_warps, num_stages=num_stages),
    )
    out = fn(x2, w, bias_arg, table)
    if pad_m:
        out = out[:m]
    return out.reshape(*x.shape[:-1], n)


def int_linear_lut_fused(module, x: jax.Array) -> jax.Array:
    """Apply an ``IntLinearLUT`` module via the fused kernel.

    Reads ``module.linear.kernel`` / ``module.linear.bias`` /
    ``module.lut.table`` — the unfused module and this function are
    interchangeable in the forward pass (bit-identical outputs).
    """
    linear = module.linear
    bias = linear.bias[...] if (linear.use_bias and linear.bias is not None) else None
    return fused_linear_lut(
        x, linear.kernel[...], module.lut.table[...], bias=bias
    )
