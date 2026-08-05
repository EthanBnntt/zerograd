"""Pure-integer helpers: quantize once, stay in int4/int8/int16/int32 thereafter."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from ._manifest import ParameterTree

Array = jax.Array

# Signed int4 stored in int8 containers (JAX has no native int4 dtype).
INT4_MIN = -8
INT4_MAX = 7

# EGG / Appendix G: symmetric int8 range (exclude -128).
EGG_I8_MIN = -127
EGG_I8_MAX = 127

# BitNet-style ternary weights {-1, 0, +1} (stored as int8).
TERNARY_MIN = -1
TERNARY_MAX = 1
ABSMAX_INT8 = 127


def egg_clip_cast(x: Array) -> Array:
    """Core EGG nonlinearity: int32 accumulator → clip → int8.

    Saturation at ±127 is the network's only intended activation.
    """
    return jnp.clip(x.astype(jnp.int32), EGG_I8_MIN, EGG_I8_MAX).astype(jnp.int8)


def egg_matmul_divisor(in_features: int) -> int:
    """EGG ``@`` scale: ``16 * √n`` (Appendix G.4), integer divisor ≥ 1."""
    if in_features < 1:
        raise ValueError(f"in_features must be positive, got {in_features}")
    return max(1, int(round(16.0 * math.sqrt(float(in_features)))))


def egg_requantize(
    y: Array,
    in_features: int,
    *,
    act_dtype: jnp.dtype = jnp.int8,
) -> Array:
    """Scale int32 matmul accum by ``16√n`` then clip-cast (EGG ``u@M``).

    When ``act_dtype`` is int32 (e.g. CIFAR logits), skip the narrow clip so
    the loss still sees a usable dynamic range.
    """
    scaled = y.astype(jnp.int32) // egg_matmul_divisor(in_features)
    if act_dtype == jnp.int32:
        return scaled
    return egg_clip_cast(scaled)


def egg_init_matrix(key: Array, shape: tuple[int, ...]) -> Array:
    """Appendix G.3: ``round(16 · N(0,1))`` clipped to EGG int8."""
    w = 16.0 * jax.random.normal(key, shape)
    return egg_clip_cast(jnp.rint(w))


def float_to_egg_i8(x: Array, *, scale: float = 16.0) -> Array:
    """Fixed-scale float → EGG int8 (no per-tensor absmax blow-up)."""
    return egg_clip_cast(jnp.rint(x.astype(jnp.float32) * float(scale)))


def factor_compute_dtype(dtype: jnp.dtype) -> jnp.dtype:
    """ES factors stay float32 when the leaf is integer (cast-to-int8 destroys signal)."""
    if jnp.issubdtype(dtype, jnp.integer):
        return jnp.float32
    return dtype


def qrange(bits: int) -> tuple[int, int]:
    """Symmetric integer range for ``bits``-wide signed values.

    ``bits=2`` is ternary ``{-1, 0, +1}`` (BitNet b1.58), not two's-complement
    int2 ``[-2, 1]``.
    """
    if bits < 2 or bits > 32:
        raise ValueError(f"bits must be in [2, 32], got {bits}")
    if bits == 2:
        return TERNARY_MIN, TERNARY_MAX
    if bits == 4:
        return INT4_MIN, INT4_MAX
    if bits in (8, 16, 32):
        info = jnp.iinfo({8: jnp.int8, 16: jnp.int16, 32: jnp.int32}[bits])
        return int(info.min), int(info.max)
    max_v = (1 << (bits - 1)) - 1
    return -max_v - 1, max_v


def absmean(x: Array, *, eps: float = 1e-5) -> Array:
    """Absmean scale ``γ = mean(|x|)`` (BitNet weight scaling), floored by ``eps``."""
    g = jnp.mean(jnp.abs(x.astype(jnp.float32)))
    return jnp.maximum(g, jnp.asarray(eps, dtype=jnp.float32))


def quantize_ternary(w: Array, *, eps: float = 1e-5) -> tuple[Array, Array]:
    """Absmean ternary quantize: ``W̃ = clip(round(W / γ), -1, 1)``, ``γ = mean(|W|)``.

    Returns ``(W̃ as int8, γ as float32 scalar)``. No shadow weights are kept —
    callers store ``W̃`` and ``γ`` directly.
    """
    w32 = w.astype(jnp.float32)
    gamma = absmean(w32, eps=eps)
    w_t = jnp.clip(jnp.rint(w32 / gamma), TERNARY_MIN, TERNARY_MAX).astype(jnp.int8)
    return w_t, gamma


def lecun_normal(key: Array, shape: tuple[int, ...]) -> Array:
    """Lecun normal in float32 (fan-in over axis 0 for 2-D kernels)."""
    if len(shape) < 2:
        return jax.random.normal(key, shape, dtype=jnp.float32)
    fan_in = int(shape[0])
    std = math.sqrt(1.0 / fan_in) if fan_in > 0 else 1.0
    return jax.random.normal(key, shape, dtype=jnp.float32) * std


def ternary_init(key: Array, shape: tuple[int, ...], *, eps: float = 1e-5) -> tuple[Array, Array]:
    """Lecun-normal → absmean ternary init. Returns ``(W̃ int8, γ float32)``."""
    return quantize_ternary(lecun_normal(key, shape), eps=eps)


def absmax_quantize_int8(x: Array, *, eps: float = 1e-5) -> tuple[Array, Array]:
    """Per-tensor absmax INT8 activation quantize (BitNet).

    ``η = max(|x|)``, ``x̃ = clip(round(x · 127 / η), -128, 127)``.
    Returns ``(x̃ as int8, η as float32 scalar)``.
    """
    x32 = x.astype(jnp.float32)
    eta = jnp.max(jnp.abs(x32))
    eta = jnp.maximum(eta, jnp.asarray(eps, dtype=jnp.float32))
    x_q = jnp.clip(jnp.rint(x32 * (float(ABSMAX_INT8) / eta)), -128, 127).astype(jnp.int8)
    return x_q, eta


def ternary_dequant_scale(gamma: Array, eta: Array) -> Array:
    """Float rescale after INT8×ternary GEMM: ``γ · η / 127``."""
    return (
        gamma.astype(jnp.float32) * eta.astype(jnp.float32) / float(ABSMAX_INT8)
    )


def rms_norm(x: Array, scale: Array, *, eps: float = 1e-6) -> Array:
    """RMSNorm over the last axis; ``scale`` broadcasts as ``[..., d]``."""
    x32 = x.astype(jnp.float32)
    rms = jnp.sqrt(jnp.mean(jnp.square(x32), axis=-1, keepdims=True) + eps)
    y = x32 / rms
    return (y * scale.astype(jnp.float32)).astype(x.dtype)


def float_to_int(x: Array, dtype: jnp.dtype = jnp.int8, *, bits: int | None = None) -> Array:
    """Symmetric per-tensor float → integer quantization (entry into int space).

    For int4, pass ``bits=4`` (values stored as ``int8`` clipped to ``[-8, 7]``).
    """
    if bits is not None:
        qmin, qmax = qrange(bits)
        storage = jnp.int8 if bits <= 8 else (jnp.int16 if bits <= 16 else jnp.int32)
    else:
        info = jnp.iinfo(dtype)
        qmin, qmax = int(info.min), int(info.max)
        storage = dtype
    max_abs = jnp.max(jnp.abs(x))
    scale = jnp.where(max_abs > 0, max_abs / qmax, jnp.asarray(1.0, x.dtype))
    return jnp.clip(jnp.rint(x / scale), qmin, qmax).astype(storage)


def float_to_int4(x: Array) -> Array:
    """Float → signed int4 stored in int8."""
    return float_to_int(x, bits=4)


def clip_int(x: Array, bits: int) -> Array:
    """Clip an integer array into the signed ``bits`` range (int4 stored as int8)."""
    qmin, qmax = qrange(bits)
    storage = jnp.int8 if bits <= 8 else x.dtype
    return jnp.clip(x.astype(jnp.int32), qmin, qmax).astype(storage)


def requantize(
    x: Array,
    shift: int,
    dtype: jnp.dtype = jnp.int8,
    *,
    bits: int | None = None,
) -> Array:
    """Right-shift int32 accumulator and clip into ``dtype`` / ``bits`` range."""
    if bits is not None:
        qmin, qmax = qrange(bits)
        storage = jnp.int8 if bits <= 8 else dtype
    else:
        info = jnp.iinfo(dtype)
        qmin, qmax = int(info.min), int(info.max)
        storage = dtype
    return jnp.clip(x.astype(jnp.int32) >> int(shift), qmin, qmax).astype(storage)


def int_conv2d(
    x: Array,
    kernel: Array,
    *,
    strides: tuple[int, int] = (1, 1),
    padding: str | tuple = "SAME",
    accum: jnp.dtype = jnp.int32,
) -> Array:
    """NHWC int convolution via im2col + :func:`int_matmul` → int32.

    ``kernel`` is ``[kH, kW, in_c, out_c]`` (or pass a pre-flattened
    ``[kH·kW·in, out]`` MATRIX with ``kernel_hw`` via :func:`int_conv2d_flat`).

    Implemented as patch-extract + int8 matmul so ROCm/CUDA stacks that lack
    portable int8 cudnn/MIOpen conv still hit the WMMA IU8 path.
    """
    if kernel.ndim == 2:
        raise ValueError("use int_conv2d_flat for a flattened MATRIX kernel")
    if x.ndim != 4 or kernel.ndim != 4:
        raise ValueError(
            f"int_conv2d expects NHWC x and [kH,kW,in,out] kernel, got {x.shape} / {kernel.shape}"
        )
    kh, kw, ic, oc = map(int, kernel.shape)
    if int(x.shape[-1]) != ic:
        raise ValueError(f"x channels {x.shape[-1]} != kernel in_features {ic}")
    return int_conv2d_flat(
        x,
        kernel.reshape(kh * kw * ic, oc),
        kernel_hw=(kh, kw),
        strides=strides,
        padding=padding,
        accum=accum,
    )


def int_conv2d_flat(
    x: Array,
    weight_2d: Array,
    *,
    kernel_hw: tuple[int, int],
    strides: tuple[int, int] = (1, 1),
    padding: str | tuple = "SAME",
    accum: jnp.dtype = jnp.int32,
) -> Array:
    """NHWC int conv with flattened MATRIX weights ``[kH·kW·in, out]``."""
    if x.ndim != 4 or weight_2d.ndim != 2:
        raise ValueError(
            f"int_conv2d_flat expects NHWC x and 2-D weight, got {x.shape} / {weight_2d.shape}"
        )
    if not jnp.issubdtype(x.dtype, jnp.integer) or not jnp.issubdtype(
        weight_2d.dtype, jnp.integer
    ):
        raise TypeError(
            f"int_conv2d_flat requires integer operands, got {x.dtype} and {weight_2d.dtype}"
        )
    kh, kw = kernel_hw
    in_c = int(x.shape[-1])
    fan_in = kh * kw * in_c
    if int(weight_2d.shape[0]) != fan_in:
        raise ValueError(
            f"weight fan-in {weight_2d.shape[0]} != {kh}*{kw}*{in_c}"
        )
    patches = jax.lax.conv_general_dilated_patches(
        x,
        filter_shape=(kh, kw),
        window_strides=strides,
        padding=padding,
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    # patches: [B, H', W', kh*kw*C]
    b, h, w, _ = patches.shape
    y = int_matmul(patches.reshape(b * h * w, fan_in), weight_2d, accum=accum)
    return y.reshape(b, h, w, int(weight_2d.shape[1]))


def int_avg_pool2d(x: Array, window: tuple[int, int], strides: tuple[int, int]) -> Array:
    """Integer average pool (truncated division); keeps input dtype."""
    if x.ndim != 4:
        raise ValueError(f"int_avg_pool2d expects NHWC, got {x.shape}")
    wh, ww = window
    sh, sw = strides
    # Reduce-window sum then // (wh*ww).
    summed = jax.lax.reduce_window(
        x.astype(jnp.int32),
        0,
        jax.lax.add,
        (1, wh, ww, 1),
        (1, sh, sw, 1),
        "VALID",
    )
    return (summed // (wh * ww)).astype(x.dtype)


def int_matmul(lhs: Array, rhs: Array, *, accum: jnp.dtype = jnp.int32) -> Array:
    """Integer matmul that keeps narrow inputs for the XLA dot (int8×int8→int32).

    Casts to int32 *before* ``@`` force an upcast path and miss RDNA4 WMMA IU8
    (``V_WMMA_I32_16X16X16_IU8``). Pass int8 (or int4-in-int8) operands and set
    ``preferred_element_type`` so the HLO is ``dot(s8, s8) -> s32``.
    """
    if not jnp.issubdtype(lhs.dtype, jnp.integer) or not jnp.issubdtype(rhs.dtype, jnp.integer):
        raise TypeError(
            f"int_matmul requires integer operands, got {lhs.dtype} and {rhs.dtype}"
        )
    # Narrow to int8 when both fit; wider ints keep their dtype for the dot.
    if lhs.dtype == jnp.int8 and rhs.dtype == jnp.int8:
        return jnp.matmul(lhs, rhs, preferred_element_type=accum)
    if (
        jnp.iinfo(lhs.dtype).bits <= 8
        and jnp.iinfo(rhs.dtype).bits <= 8
        and lhs.dtype != jnp.bool_
        and rhs.dtype != jnp.bool_
    ):
        return jnp.matmul(
            lhs.astype(jnp.int8),
            rhs.astype(jnp.int8),
            preferred_element_type=accum,
        )
    return jnp.matmul(lhs, rhs, preferred_element_type=accum)


def dyadic_mul(a: Array, b: Array, *, mult: int = 1, shift: int = 3, bits: int = 4) -> Array:
    """Element-wise int product with dyadic rescaling: ``(a * b * mult) >> shift``."""
    y = a.astype(jnp.int32) * b.astype(jnp.int32) * int(mult)
    return requantize(y, shift, bits=bits)


def int_relu(x: Array) -> Array:
    """Integer ReLU (keeps input dtype)."""
    return jnp.maximum(x, jnp.zeros((), dtype=x.dtype))


def int_mean(x: Array, axis: int = -1, keepdims: bool = True) -> Array:
    """Integer mean via truncated division (no float mean)."""
    x32 = x.astype(jnp.int32)
    n = x32.shape[axis]
    return (jnp.sum(x32, axis=axis, keepdims=keepdims) // n).astype(x.dtype)


# int4 GELU LUT for indices -8..7 (approx gelu mapped back into int4).
_GELU_INT4_LUT = jnp.asarray(
    [-8, -8, -7, -6, -4, -2, -1, 0, 0, 1, 2, 3, 4, 5, 6, 7],
    dtype=jnp.int8,
)


def int_gelu_lut(x: Array, *, bits: int = 4) -> Array:
    """GELU via LUT. For int4 inputs, indexes ``[-8, 7]`` → LUT."""
    if bits == 4:
        idx = jnp.clip(x.astype(jnp.int32) - INT4_MIN, 0, 15)
        return _GELU_INT4_LUT[idx]
    return int_relu(x)


# Precomputed ≈1024 * exp(k/4) for k in [-32..0] (33 entries).
_SOFTMAX_LUT = jnp.asarray(
    [
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        2,
        2,
        3,
        4,
        5,
        6,
        8,
        10,
        13,
        16,
        21,
        26,
        33,
        42,
        54,
        68,
        86,
        109,
        139,
        176,
        223,
        283,
        359,
        455,
        577,
        1024,
    ],
    dtype=jnp.int32,
)


def int_softmax(scores: Array, *, bits: int = 8) -> Array:
    """Integer softmax via LUT on ``(x - max)`` clipped to ``[-32, 0]``."""
    s32 = scores.astype(jnp.int32)
    m = jnp.max(s32, axis=-1, keepdims=True)
    z = jnp.clip(s32 - m, -32, 0)
    idx = (z + 32).astype(jnp.int32)
    exps = _SOFTMAX_LUT[idx]
    total = jnp.sum(exps, axis=-1, keepdims=True)
    total = jnp.maximum(total, 1)
    return ((exps << bits) // total).astype(jnp.int16)


def snap_tree_to_integer(
    updated: Array | ParameterTree,
    template: Array | ParameterTree,
    *,
    bits: int | None = None,
) -> Array | ParameterTree:
    """Round/clip Optax float updates back onto the template's integer dtypes.

    When ``bits`` is set (e.g. 4), int8 leaves are clipped to that signed range
    (int4-in-int8); wider integer leaves still use their dtype iinfo.
    """
    if isinstance(updated, dict) and isinstance(template, dict):
        return {
            k: snap_tree_to_integer(updated[k], template[k], bits=bits) for k in template
        }
    if isinstance(template, jax.Array) and jnp.issubdtype(template.dtype, jnp.integer):
        info = jnp.iinfo(template.dtype)
        if bits is not None and template.dtype == jnp.int8:
            lo, hi = qrange(bits)
        else:
            lo, hi = int(info.min), int(info.max)
        assert isinstance(updated, jax.Array)
        return jnp.clip(jnp.rint(updated), lo, hi).astype(template.dtype)
    return updated


def float_view_tree(params: Array | ParameterTree) -> Array | ParameterTree:
    """Cast integer leaves to float32 for Optax state / apply_updates."""
    if isinstance(params, dict):
        return {k: float_view_tree(v) for k, v in params.items()}
    if isinstance(params, jax.Array) and jnp.issubdtype(params.dtype, jnp.integer):
        return params.astype(jnp.float32)
    return params
