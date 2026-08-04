"""Flax NNX surgery: factor-aware layers and auto-manifest construction.

ZeroGrad never materializes dense ``W + A @ B`` deltas. Instead, ``init`` walks
an ``nnx.Module`` graph, replaces Linear / Embed / LayerNorm / bare Params with
surged modules that call :mod:`zerograd._candidate` factor ops when a candidate
is bound on the shared :class:`ZeroGradSlot`, and builds a :class:`Manifest`
from stable graph paths for deterministic replay.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from ._candidate import (
    perturbed_conv,
    perturbed_int_conv,
    perturbed_int_linear,
    perturbed_int_table_lookup,
    perturbed_int_table_lookup_prepared,
    perturbed_int_vector,
    perturbed_linear,
    perturbed_table_lookup,
    perturbed_table_lookup_prepared,
    perturbed_tied_logits,
    perturbed_vector,
)
from ._integer import (
    EGG_I8_MAX,
    EGG_I8_MIN,
    absmax_quantize_int8,
    egg_init_matrix,
    egg_matmul_divisor,
    egg_requantize,
    float_to_int,
    int_conv2d,
    int_conv2d_flat,
    int_matmul,
    qrange,
    requantize,
    rms_norm,
    ternary_dequant_scale,
    ternary_init,
)
from ._keys import group_key
from ._manifest import Manifest, ManifestEntry, ParameterLayout

Array = jax.Array

LAYOUT_METADATA_KEY = "zerograd_layout"


def mark_table(param: nnx.Param) -> nnx.Param:
    """Tag a 2-D ``nnx.Param`` as a TABLE (embedding-style) for auto-manifest."""
    if not isinstance(param, nnx.Param):
        raise TypeError("mark_table expects an nnx.Param")
    param.set_metadata(**{LAYOUT_METADATA_KEY: ParameterLayout.TABLE.value})
    return param


def _layout_from_param(param: nnx.Param) -> ParameterLayout:
    if param.has_metadata(LAYOUT_METADATA_KEY):
        raw = param.get_metadata(LAYOUT_METADATA_KEY)
        return ParameterLayout(raw)
    value = param[...]
    if value.ndim == 1:
        return ParameterLayout.VECTOR
    if value.ndim == 2:
        return ParameterLayout.MATRIX
    raise ValueError(
        f"ZeroGrad only supports 1-D or 2-D parameters, got shape {value.shape}"
    )


def _path_str(path: tuple[Any, ...]) -> str:
    return ".".join(str(p) for p in path)


def _path_tuple(path: tuple[Any, ...]) -> tuple[str, ...]:
    """Manifest paths must be non-empty strings (nnx.List uses int indices)."""
    return tuple(str(p) for p in path)


class ZeroGradSlot(nnx.Module):
    """Shared candidate binding for all surged layers on one model."""

    def __init__(
        self,
        rank: int = 1,
        sigma: float = 0.01,
        *,
        sigma_shift: int = 4,
        integer_es: bool = False,
    ) -> None:
        self.key = nnx.Variable(jax.random.key(0))
        self.enabled: bool = False
        self.rank: int = rank
        self.sigma: float = sigma
        self.sigma_shift: int = int(sigma_shift)
        self.integer_es: bool = bool(integer_es)
        self.factor_sign = nnx.Variable(jnp.int32(1))
        self.manifest: Manifest | None = None


def _factor_key(slot: ZeroGradSlot, group: str) -> Array:
    manifest = slot.manifest
    if manifest is None:
        raise RuntimeError("ZeroGradSlot.manifest is not set; call ZeroGrad.init first")
    return group_key(slot.key[...], manifest, group)


class ZgLinear(nnx.Module):
    """``nnx.Linear`` replacement with factor-only matrix/bias perturbations."""

    def __init__(
        self,
        linear: nnx.Linear,
        slot: ZeroGradSlot,
        kernel_group: str,
        bias_group: str | None,
    ) -> None:
        self.kernel = linear.kernel
        self.bias = linear.bias
        self.use_bias = linear.use_bias
        self.slot = slot
        self.kernel_group = kernel_group
        self.bias_group = bias_group

    def __call__(self, x: Array) -> Array:
        slot = self.slot
        kernel = self.kernel[...]
        if slot.enabled:
            y = perturbed_linear(
                x,
                kernel,
                _factor_key(slot, self.kernel_group),
                slot.rank,
                slot.sigma,
                factor_sign=slot.factor_sign[...],
            )
        else:
            y = x @ kernel
        if self.use_bias and self.bias is not None:
            bias = self.bias[...]
            if slot.enabled and self.bias_group is not None:
                bias = perturbed_vector(
                    bias,
                    _factor_key(slot, self.bias_group),
                    slot.sigma,
                    factor_sign=slot.factor_sign[...],
                )
            y = y + bias
        return y


class ZgConv(nnx.Module):
    """``nnx.Conv`` replacement; kernel stored as ``[kH·kW·in, out]`` MATRIX."""

    def __init__(
        self,
        conv: nnx.Conv,
        slot: ZeroGradSlot,
        kernel_group: str,
        bias_group: str | None,
    ) -> None:
        k = conv.kernel[...]
        if k.ndim != 4:
            raise ValueError(f"ZgConv expects a 2-D conv kernel, got shape {k.shape}")
        kh, kw, ic, oc = map(int, k.shape)
        self.kernel_hw = (kh, kw)
        self.in_features = ic
        self.out_features = oc
        # Flatten so ZeroGrad MATRIX replay / factors stay 2-D.
        self.kernel = nnx.Param(k.reshape(kh * kw * ic, oc))
        self.bias = conv.bias
        self.use_bias = conv.use_bias
        self.strides = conv.strides
        self.padding = conv.padding
        self.input_dilation = conv.input_dilation
        self.kernel_dilation = conv.kernel_dilation
        self.feature_group_count = conv.feature_group_count
        self.precision = conv.precision
        self.dtype = conv.dtype
        self.conv_general_dilated = conv.conv_general_dilated
        self.promote_dtype = conv.promote_dtype
        self.preferred_element_type = conv.preferred_element_type
        self.slot = slot
        self.kernel_group = kernel_group
        self.bias_group = bias_group

    def _maybe_broadcast(self, x: int | tuple[int, ...] | None) -> tuple[int, ...]:
        nd = len(self.kernel_hw)
        if x is None:
            x = 1
        if isinstance(x, int):
            return (x,) * nd
        return tuple(x)

    def __call__(self, x: Array) -> Array:
        from flax.nnx.nn.linear import _conv_dimension_numbers, canonicalize_padding

        slot = self.slot
        kh, kw = self.kernel_hw
        kernel_2d = self.kernel[...]
        strides = self._maybe_broadcast(self.strides)
        input_dilation = self._maybe_broadcast(self.input_dilation)
        kernel_dilation = self._maybe_broadcast(self.kernel_dilation)
        padding_lax = canonicalize_padding(self.padding, len(self.kernel_hw))
        if padding_lax in ("CIRCULAR", "REFLECT", "CAUSAL"):
            raise NotImplementedError(
                f"ZgConv does not support padding={padding_lax!r}; use SAME/VALID"
            )

        conv_kwargs: dict[str, Any] = {}
        if self.preferred_element_type is not None:
            conv_kwargs["preferred_element_type"] = self.preferred_element_type

        def _conv(inputs: Array, kernel4: Array) -> Array:
            inputs_p, kernel_p, _ = self.promote_dtype(
                (inputs, kernel4, None), dtype=self.dtype
            )
            return self.conv_general_dilated(
                inputs_p,
                kernel_p,
                strides,
                padding_lax,
                lhs_dilation=input_dilation,
                rhs_dilation=kernel_dilation,
                dimension_numbers=_conv_dimension_numbers(inputs_p.shape),
                feature_group_count=self.feature_group_count,
                precision=self.precision,
                **conv_kwargs,
            )

        if slot.enabled:
            x_p, k_p, _ = self.promote_dtype(
                (x, kernel_2d, None), dtype=self.dtype
            )
            y = perturbed_conv(
                x_p,
                k_p,
                _factor_key(slot, self.kernel_group),
                slot.rank,
                slot.sigma,
                kernel_hw=self.kernel_hw,
                in_features=self.in_features,
                strides=strides,
                padding=padding_lax,
                lhs_dilation=input_dilation,
                rhs_dilation=kernel_dilation,
                feature_group_count=self.feature_group_count,
                precision=self.precision,
                factor_sign=slot.factor_sign[...],
                conv_general_dilated=self.conv_general_dilated,
            )
        else:
            y = _conv(x, kernel_2d.reshape(kh, kw, self.in_features, self.out_features))

        if self.use_bias and self.bias is not None:
            bias = self.bias[...]
            if slot.enabled and self.bias_group is not None:
                bias = perturbed_vector(
                    bias,
                    _factor_key(slot, self.bias_group),
                    slot.sigma,
                    factor_sign=slot.factor_sign[...],
                )
            y = y + bias.reshape((1,) * (y.ndim - 1) + bias.shape)
        return y


class TernaryLinear(nnx.Module):
    """BitNet-style linear: bf16 → RMSNorm → INT8 × ternary → bf16.

    Weights are stored **directly** as ``{-1, 0, +1}`` (int8) with a float
    absmean scale ``γ`` — no FP shadow weights. Activations stay bf16 on the
    residual highway; only the GEMM runs as INT8×ternary→INT32, then rescales
    by ``γ · η / 127``.

    ZeroGrad perturbs the ternary kernel via integer factor ops and ``γ`` /
    bias / RMS scale via float factor ops (see :class:`ZgTernaryLinear`).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        use_bias: bool = True,
        use_rms_norm: bool = True,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs,
    ) -> None:
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.use_bias = bool(use_bias)
        self.use_rms_norm = bool(use_rms_norm)
        self.dtype = dtype
        key = rngs.params()
        w_t, gamma = ternary_init(key, (in_features, out_features))
        self.kernel = nnx.Param(w_t)
        # Shape (1,) so surgery wraps it as ZgVector (1-D).
        self.gamma = nnx.Param(jnp.reshape(gamma, (1,)).astype(jnp.float32))
        self.rms_scale = (
            nnx.Param(jnp.ones((in_features,), dtype=jnp.float32))
            if use_rms_norm
            else None
        )
        self.bias = (
            nnx.Param(jnp.zeros((out_features,), dtype=dtype)) if use_bias else None
        )

    def __call__(self, x: Array) -> Array:
        return _ternary_linear_forward(
            x.astype(self.dtype),
            self.kernel[...],
            self.gamma[...],
            rms_scale=self.rms_scale[...] if self.rms_scale is not None else None,
            bias=self.bias[...] if self.bias is not None else None,
            use_rms_norm=self.use_rms_norm,
        )


def _ternary_linear_forward(
    x: Array,
    kernel: Array,
    gamma: Array,
    *,
    rms_scale: Array | None,
    bias: Array | None,
    use_rms_norm: bool,
    key: Array | None = None,
    rank: int = 1,
    sigma_shift: int = 4,
    factor_sign: Array | int = 1,
    enabled: bool = False,
    gamma_key: Array | None = None,
    rms_key: Array | None = None,
    bias_key: Array | None = None,
    sigma: float = 0.01,
) -> Array:
    """Shared TernaryLinear / ZgTernaryLinear forward (bf16 highway, int GEMM)."""
    h = x
    if use_rms_norm and rms_scale is not None:
        scale = rms_scale
        if enabled and rms_key is not None:
            scale = perturbed_vector(scale, rms_key, sigma, factor_sign=factor_sign)
        h = rms_norm(h, scale)
    x_q, eta = absmax_quantize_int8(h)
    if enabled and key is not None:
        y32 = perturbed_int_linear(
            x_q, kernel, key, rank, sigma_shift, factor_sign=factor_sign
        )
    else:
        y32 = int_matmul(x_q, kernel)
    g = gamma
    if enabled and gamma_key is not None:
        g = perturbed_vector(g, gamma_key, sigma, factor_sign=factor_sign)
    # Absmean scale must stay positive after ES updates.
    g_pos = jnp.maximum(jnp.abs(jnp.reshape(g, ())), jnp.asarray(1e-5, dtype=jnp.float32))
    scale = ternary_dequant_scale(g_pos, eta)
    y = y32.astype(jnp.float32) * scale
    y = y.astype(x.dtype)
    if bias is not None:
        b = bias
        if enabled and bias_key is not None:
            b = perturbed_vector(b, bias_key, sigma, factor_sign=factor_sign)
        y = y + b.astype(y.dtype)
    return y


class ZgTernaryLinear(nnx.Module):
    """``TernaryLinear`` with factor-only ternary / float perturbations."""

    def __init__(
        self,
        linear: TernaryLinear,
        slot: ZeroGradSlot,
        kernel_group: str,
        gamma_group: str,
        rms_group: str | None,
        bias_group: str | None,
    ) -> None:
        self.kernel = linear.kernel
        self.gamma = linear.gamma
        self.rms_scale = linear.rms_scale
        self.bias = linear.bias
        self.use_bias = linear.use_bias
        self.use_rms_norm = linear.use_rms_norm
        self.dtype = linear.dtype
        self.slot = slot
        self.kernel_group = kernel_group
        self.gamma_group = gamma_group
        self.rms_group = rms_group
        self.bias_group = bias_group

    def __call__(self, x: Array) -> Array:
        slot = self.slot
        enabled = slot.enabled
        k_key = _factor_key(slot, self.kernel_group) if enabled else None
        g_key = _factor_key(slot, self.gamma_group) if enabled else None
        r_key = (
            _factor_key(slot, self.rms_group)
            if enabled and self.rms_group is not None
            else None
        )
        b_key = (
            _factor_key(slot, self.bias_group)
            if enabled and self.bias_group is not None
            else None
        )
        return _ternary_linear_forward(
            x.astype(self.dtype),
            self.kernel[...],
            self.gamma[...],
            rms_scale=self.rms_scale[...] if self.rms_scale is not None else None,
            bias=self.bias[...] if self.bias is not None else None,
            use_rms_norm=self.use_rms_norm,
            key=k_key,
            rank=slot.rank,
            sigma_shift=slot.sigma_shift,
            factor_sign=slot.factor_sign[...] if enabled else 1,
            enabled=enabled,
            gamma_key=g_key,
            rms_key=r_key,
            bias_key=b_key,
            sigma=slot.sigma,
        )


class IntLinear(nnx.Module):
    """Pure-integer linear: int activations × int weights → int32 → requantize.

    Weights are stored as true integers (int4-in-int8, int8, …). Float is only
    used to initialize via :func:`float_to_int` unless ``egg=True``.

    With ``egg=True`` (Appendix G): init ``round(16·N(0,1))``, scale accum by
    ``16√n``, then clip-cast to int8 — saturation is the only nonlinearity.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        use_bias: bool = True,
        bits: int = 8,
        w_dtype: jnp.dtype | None = None,
        act_dtype: jnp.dtype | None = None,
        out_shift: int = 8,
        egg: bool = False,
        rngs: nnx.Rngs,
    ) -> None:
        self.bits = int(bits)
        self.in_features = int(in_features)
        self.egg = bool(egg)
        if w_dtype is None:
            w_dtype = jnp.int8 if bits <= 8 else (jnp.int16 if bits <= 16 else jnp.int32)
        if act_dtype is None:
            act_dtype = w_dtype
        if self.egg:
            self.kernel = nnx.Param(egg_init_matrix(rngs.params(), (in_features, out_features)))
        else:
            w = nnx.initializers.lecun_normal()(rngs.params(), (in_features, out_features))
            self.kernel = nnx.Param(float_to_int(w, w_dtype, bits=bits))
        self.bias = (
            nnx.Param(jnp.zeros((out_features,), dtype=jnp.int32)) if use_bias else None
        )
        self.use_bias = use_bias
        self.out_shift = int(out_shift)
        self.act_dtype = act_dtype
        self.w_dtype = w_dtype

    def __call__(self, x: Array) -> Array:
        y = int_matmul(x, self.kernel[...])
        if self.use_bias and self.bias is not None:
            y = y + self.bias[...]
        if self.egg:
            return egg_requantize(y, self.in_features, act_dtype=self.act_dtype)
        # Wider act_dtype (e.g. int32 logits) skips narrow bit clipping.
        rq_bits = None if self.act_dtype == jnp.int32 else self.bits
        return requantize(y, self.out_shift, self.act_dtype, bits=rq_bits)


class IntLUT(nnx.Module):
    """Element-wise int8→int8 nonlinearity via a learnable 256-entry LUT.

    Default ``init=\"identity\"`` is the linear map ``f(v)=v`` for every
    signed int8 value (``table[i] = i - 128``, so ``-128→-128``, …, ``127→127``).
    ZeroGrad surgery replaces this with :class:`ZgIntLUT`, which perturbs and
    updates ``table`` so ES can learn a better pointwise function.
    """

    def __init__(
        self,
        *,
        init: str = "identity",
        explore_shift: int = 0,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        del rngs
        if not isinstance(explore_shift, int) or isinstance(explore_shift, bool) or explore_shift < 0:
            raise ValueError(f"explore_shift must be a non-negative int, got {explore_shift!r}")
        domain = jnp.arange(-128, 128, dtype=jnp.int32)
        if init == "identity":
            # Linear / identity: table[x + 128] == x for all int8 x.
            values = domain
        elif init == "relu":
            values = jnp.maximum(domain, 0)
        elif init == "egg_clip":
            values = jnp.clip(domain, EGG_I8_MIN, EGG_I8_MAX)
        else:
            raise ValueError(f"unknown IntLUT init {init!r}; use identity|relu|egg_clip")
        self.table = nnx.Param(values.astype(jnp.int8))
        # σ̂ for LUT vector noise only (total scale 2^{-(4+explore_shift)}).
        # Default 0 → ±1…3 exploration so the table can move under ES.
        self.explore_shift = int(explore_shift)

    def __call__(self, x: Array) -> Array:
        lut = self.table() if isinstance(self.table, ZgVector) else self.table[...]
        idx = x.astype(jnp.int32) - jnp.iinfo(jnp.int8).min
        return lut[idx]


class IntLinearLUT(nnx.Module):
    """Core int8 block: ``int8 @ int8 → int32 → int8`` then learnable ``IntLUT``.

    This is the main primitive for pure-integer networks: a GEMM / linear that
    stays in int8 via EGG (or shift) requantize, followed by a pointwise
    256-entry LUT nonlinearity. Surgery walks into ``linear`` / ``lut`` and
    wraps them as :class:`ZgIntLinear` / :class:`ZgIntLUT`.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        use_bias: bool = False,
        bits: int = 8,
        out_shift: int = 8,
        egg: bool = True,
        lut_init: str = "identity",
        explore_shift: int = 0,
        rngs: nnx.Rngs,
    ) -> None:
        self.linear = IntLinear(
            in_features,
            out_features,
            use_bias=use_bias,
            bits=bits,
            out_shift=out_shift,
            egg=egg,
            act_dtype=jnp.int8,
            rngs=rngs,
        )
        self.lut = IntLUT(init=lut_init, explore_shift=explore_shift, rngs=rngs)

    def __call__(self, x: Array) -> Array:
        return self.lut(self.linear(x))


class ZgIntLUT(nnx.Module):
    """``IntLUT`` with factor-only int8 vector perturbations on the table."""

    def __init__(self, lut: IntLUT, slot: ZeroGradSlot, group: str) -> None:
        self.table = lut.table
        self.explore_shift = int(lut.explore_shift)
        self.slot = slot
        self.group = group

    def __call__(self, x: Array) -> Array:
        lut = self.table[...]
        slot = self.slot
        if slot.enabled:
            lut = perturbed_int_vector(
                lut,
                _factor_key(slot, self.group),
                self.explore_shift,
                factor_sign=slot.factor_sign[...],
            )
        idx = x.astype(jnp.int32) - jnp.iinfo(jnp.int8).min
        return lut[idx]


class IntConv(nnx.Module):
    """Pure-integer 2-D conv: int8 NHWC × int8 ``[kH·kW·in, out]`` → requantize.

    Kernel is stored flattened as a MATRIX leaf so ZeroGrad factor replay matches
    :class:`IntLinear`. With ``egg=True``: EGG init + ``16√n`` scale + clip-cast.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int | tuple[int, int] = 3,
        *,
        strides: int | tuple[int, int] = 1,
        padding: str = "SAME",
        use_bias: bool = True,
        bits: int = 8,
        out_shift: int = 8,
        egg: bool = False,
        rngs: nnx.Rngs,
    ) -> None:
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(strides, int):
            strides = (strides, strides)
        kh, kw = int(kernel_size[0]), int(kernel_size[1])
        self.kernel_hw = (kh, kw)
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.strides = (int(strides[0]), int(strides[1]))
        self.padding = padding
        self.bits = int(bits)
        self.out_shift = int(out_shift)
        self.egg = bool(egg)
        self.use_bias = bool(use_bias)
        fan_in = kh * kw * self.in_features
        if self.egg:
            self.kernel = nnx.Param(egg_init_matrix(rngs.params(), (fan_in, out_features)))
        else:
            w = nnx.initializers.lecun_normal()(rngs.params(), (fan_in, out_features))
            self.kernel = nnx.Param(float_to_int(w, jnp.int8, bits=bits))
        self.bias = (
            nnx.Param(jnp.zeros((out_features,), dtype=jnp.int32)) if use_bias else None
        )

    def __call__(self, x: Array) -> Array:
        kh, kw = self.kernel_hw
        y = int_conv2d_flat(
            x,
            self.kernel[...],
            kernel_hw=self.kernel_hw,
            strides=self.strides,
            padding=self.padding,
        )
        if self.use_bias and self.bias is not None:
            y = y + self.bias[...].reshape((1, 1, 1, -1))
        fan_in = kh * kw * self.in_features
        if self.egg:
            return egg_requantize(y, fan_in, act_dtype=jnp.int8)
        return requantize(y, self.out_shift, jnp.int8, bits=self.bits)


class ZgIntConv(nnx.Module):
    """``IntConv`` with factor-only integer convolution perturbations."""

    def __init__(
        self,
        conv: IntConv,
        slot: ZeroGradSlot,
        kernel_group: str,
        bias_group: str | None,
    ) -> None:
        self.kernel = conv.kernel
        self.bias = conv.bias
        self.kernel_hw = conv.kernel_hw
        self.in_features = conv.in_features
        self.out_features = conv.out_features
        self.strides = conv.strides
        self.padding = conv.padding
        self.bits = conv.bits
        self.out_shift = conv.out_shift
        self.egg = conv.egg
        self.use_bias = conv.use_bias
        self.slot = slot
        self.kernel_group = kernel_group
        self.bias_group = bias_group

    def __call__(self, x: Array) -> Array:
        slot = self.slot
        kh, kw = self.kernel_hw
        kernel = self.kernel[...]
        if slot.enabled:
            y = perturbed_int_conv(
                x,
                kernel,
                _factor_key(slot, self.kernel_group),
                slot.rank,
                slot.sigma_shift,
                kernel_hw=self.kernel_hw,
                in_features=self.in_features,
                strides=self.strides,
                padding=self.padding,
                factor_sign=slot.factor_sign[...],
            )
        else:
            y = int_conv2d_flat(
                x,
                kernel,
                kernel_hw=self.kernel_hw,
                strides=self.strides,
                padding=self.padding,
            )
        if self.use_bias and self.bias is not None:
            bias = self.bias[...]
            if slot.enabled and self.bias_group is not None:
                bias = perturbed_int_vector(
                    bias,
                    _factor_key(slot, self.bias_group),
                    slot.sigma_shift,
                    factor_sign=slot.factor_sign[...],
                )
            y = y + bias.astype(jnp.int32).reshape((1, 1, 1, -1))
        fan_in = kh * kw * self.in_features
        if self.egg:
            return egg_requantize(y, fan_in, act_dtype=jnp.int8)
        return requantize(y, self.out_shift, jnp.int8, bits=self.bits)


class ZgIntLinear(nnx.Module):
    """``IntLinear`` replacement with factor-only integer matmul perturbations."""

    def __init__(
        self,
        linear: IntLinear,
        slot: ZeroGradSlot,
        kernel_group: str,
        bias_group: str | None,
    ) -> None:
        self.kernel = linear.kernel
        self.bias = linear.bias
        self.use_bias = linear.use_bias
        self.out_shift = linear.out_shift
        self.act_dtype = linear.act_dtype
        self.bits = linear.bits
        self.in_features = linear.in_features
        self.egg = linear.egg
        self.slot = slot
        self.kernel_group = kernel_group
        self.bias_group = bias_group

    def __call__(self, x: Array) -> Array:
        slot = self.slot
        kernel = self.kernel[...]
        if slot.enabled:
            y = perturbed_int_linear(
                x,
                kernel,
                _factor_key(slot, self.kernel_group),
                slot.rank,
                slot.sigma_shift,
                factor_sign=slot.factor_sign[...],
            )
        else:
            y = int_matmul(x, kernel)
        if self.use_bias and self.bias is not None:
            bias = self.bias[...]
            if slot.enabled and self.bias_group is not None:
                # Integer leaves → int factors; float leaves stay on float noise
                # even when integer_es=True (mixed ternary + scale models).
                if jnp.issubdtype(bias.dtype, jnp.integer):
                    bias = perturbed_int_vector(
                        bias,
                        _factor_key(slot, self.bias_group),
                        slot.sigma_shift,
                        factor_sign=slot.factor_sign[...],
                    )
                else:
                    bias = perturbed_vector(
                        bias,
                        _factor_key(slot, self.bias_group),
                        slot.sigma,
                        factor_sign=slot.factor_sign[...],
                    )
            y = y + bias.astype(jnp.int32)
        if self.egg:
            return egg_requantize(y, self.in_features, act_dtype=self.act_dtype)
        rq_bits = None if self.act_dtype == jnp.int32 else self.bits
        return requantize(y, self.out_shift, self.act_dtype, bits=rq_bits)


class IntSpatialProj(nnx.Module):
    """Token-axis spatial projection for gMLP SGU: ``y[b,:,c] = W @ x[b,:,c] + b``.

    ``W`` is ``[seq, seq]`` int4/int8. Applied as ``flat @ W`` over reshaped
    ``[B*C, S]`` so ZeroGrad factor ops match :class:`IntLinear`.

    With ``egg=True``: EGG scaled matmul + clip-cast; zero ``W`` and bias
    ``= 16√S`` so the gate starts near multiply-by-one.
    """

    def __init__(
        self,
        seq_len: int,
        *,
        bits: int = 4,
        out_shift: int = 9,
        egg: bool = False,
        rngs: nnx.Rngs,
    ) -> None:
        self.bits = int(bits)
        self.seq_len = int(seq_len)
        self.out_shift = int(out_shift)
        self.egg = bool(egg)
        del rngs  # deterministic near-identity init; ES learns the spatial mix
        storage = jnp.int8 if bits <= 8 else (jnp.int16 if bits <= 16 else jnp.int32)
        self.kernel = nnx.Param(jnp.zeros((seq_len, seq_len), dtype=storage))
        if self.egg:
            # After // (16√S) → 1, so SGU starts as channel gate ≈ identity.
            ones_bias = egg_matmul_divisor(seq_len)
            self.bias = nnx.Param(jnp.full((seq_len,), ones_bias, dtype=jnp.int32))
        else:
            # bias >> out_shift ≈ 1 so SGU starts as channel gating ≈ identity.
            self.bias = nnx.Param(
                jnp.full((seq_len,), 1 << int(out_shift), dtype=jnp.int32)
            )

    def __call__(self, x: Array) -> Array:
        """``x``: ``[B, S, C]`` → same shape, int activations."""
        return _spatial_proj_forward(
            x,
            self.kernel[...],
            self.bias[...],
            out_shift=self.out_shift,
            bits=self.bits,
            egg=self.egg,
            in_features=self.seq_len,
        )


def _spatial_proj_forward(
    x: Array,
    kernel: Array,
    bias: Array,
    *,
    out_shift: int,
    bits: int,
    egg: bool = False,
    in_features: int | None = None,
    key: Array | None = None,
    rank: int = 1,
    sigma_shift: int = 4,
    factor_sign: Array | int = 1,
    enabled: bool = False,
) -> Array:
    b, s, c = x.shape
    flat = jnp.transpose(x, (0, 2, 1)).reshape(b * c, s)
    if enabled and key is not None:
        y = perturbed_int_linear(
            flat, kernel, key, rank, sigma_shift, factor_sign=factor_sign
        )
    else:
        y = int_matmul(flat, kernel)
    if enabled and key is not None:
        bias_key = jax.random.fold_in(key, 1)
        bias = perturbed_int_vector(bias, bias_key, sigma_shift, factor_sign=factor_sign)
    y = y + bias.astype(jnp.int32)
    if egg:
        n = int(in_features) if in_features is not None else s
        y = egg_requantize(y, n, act_dtype=jnp.int8)
    else:
        y = requantize(y, out_shift, bits=bits)
    return jnp.transpose(y.reshape(b, c, s), (0, 2, 1))


class ZgIntSpatialProj(nnx.Module):
    """``IntSpatialProj`` with Appendix H factor perturbations on ``W`` / bias."""

    def __init__(
        self,
        spatial: IntSpatialProj,
        slot: ZeroGradSlot,
        kernel_group: str,
        bias_group: str,
    ) -> None:
        self.kernel = spatial.kernel
        self.bias = spatial.bias
        self.bits = spatial.bits
        self.seq_len = spatial.seq_len
        self.out_shift = spatial.out_shift
        self.egg = spatial.egg
        self.slot = slot
        self.kernel_group = kernel_group
        self.bias_group = bias_group

    def __call__(self, x: Array) -> Array:
        slot = self.slot
        b, s, c = x.shape
        flat = jnp.transpose(x, (0, 2, 1)).reshape(b * c, s)
        kernel = self.kernel[...]
        if slot.enabled:
            y = perturbed_int_linear(
                flat,
                kernel,
                _factor_key(slot, self.kernel_group),
                slot.rank,
                slot.sigma_shift,
                factor_sign=slot.factor_sign[...],
            )
        else:
            y = int_matmul(flat, kernel)
        bias = self.bias[...]
        if slot.enabled:
            bias = perturbed_int_vector(
                bias,
                _factor_key(slot, self.bias_group),
                slot.sigma_shift,
                factor_sign=slot.factor_sign[...],
            )
        y = y + bias.astype(jnp.int32)
        if self.egg:
            y = egg_requantize(y, self.seq_len, act_dtype=jnp.int8)
        else:
            y = requantize(y, self.out_shift, bits=self.bits)
        return jnp.transpose(y.reshape(b, c, s), (0, 2, 1))


class IntAffine(nnx.Module):
    """Integer centering + learned scale/bias (LayerNorm stand-in without float).

    ``y = ((x - mean(x)) * scale) >> shift + bias`` with int16 scale, int activation bias.

    With ``egg=True``: scale init 16 (Appendix G.3 ``θln``), shift 0, clip to ±127
    so saturation remains the nonlinearity.
    """

    def __init__(
        self,
        dim: int,
        *,
        shift: int = 8,
        bits: int = 8,
        egg: bool = False,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        del rngs
        self.bits = int(bits)
        self.egg = bool(egg)
        if self.egg:
            from ._integer import EGG_I8_MAX, EGG_I8_MIN

            self.scale = nnx.Param(jnp.full((dim,), 16, dtype=jnp.int16))
            self.shift = 0
            self._qmin, self._qmax = EGG_I8_MIN, EGG_I8_MAX
            storage = jnp.int8
        else:
            self.scale = nnx.Param(jnp.full((dim,), 1 << shift, dtype=jnp.int16))
            self.shift = int(shift)
            qmin, qmax = qrange(bits)
            self._qmin, self._qmax = qmin, qmax
            storage = jnp.int8 if bits <= 8 else jnp.int16
        self.bias = nnx.Param(jnp.zeros((dim,), dtype=storage))

    def __call__(self, x: Array) -> Array:
        from ._integer import int_mean

        scale = self.scale() if isinstance(self.scale, ZgVector) else self.scale[...]
        bias = self.bias() if isinstance(self.bias, ZgVector) else self.bias[...]
        mean = int_mean(x, axis=-1, keepdims=True)
        centered = x.astype(jnp.int32) - mean.astype(jnp.int32)
        y = (centered * scale.astype(jnp.int32)) >> self.shift
        return jnp.clip(y + bias.astype(jnp.int32), self._qmin, self._qmax).astype(
            bias.dtype
        )


class ZgEmbed(nnx.Module):
    """``nnx.Embed`` replacement with factor-only table perturbations.

    Integer tables use Appendix H row-sparse gathers (no full ``A[V,r]`` alloc).
    """

    def __init__(self, embed: nnx.Embed, slot: ZeroGradSlot, group: str) -> None:
        self.embedding = embed.embedding
        self.num_embeddings = embed.num_embeddings
        self.features = embed.features
        self.slot = slot
        self.group = group

    def __call__(self, indices: Array) -> Array:
        return self.lookup(indices, factors=None)

    def candidate_factors(self) -> tuple[Array, Array] | None:
        """Generate this candidate's table factors once for reuse by a loss.

        Returns ``None`` when candidate perturbations are disabled.
        """
        table = self.embedding[...]
        slot = self.slot
        if not slot.enabled:
            return None
        if jnp.issubdtype(table.dtype, jnp.integer):
            from ._factors import int_table_factors

            return int_table_factors(
                _factor_key(slot, self.group), table.shape, slot.rank
            )
        from ._factors import table_factors

        return table_factors(
            _factor_key(slot, self.group),
            table.shape,
            slot.rank,
            dtype=table.dtype,
        )

    def lookup(
        self,
        indices: Array,
        *,
        factors: tuple[Array, Array] | None,
    ) -> Array:
        """Gather rows, optionally reusing :meth:`candidate_factors` output."""
        table = self.embedding[...]
        slot = self.slot
        if not slot.enabled:
            return table[indices]
        if factors is None:
            if jnp.issubdtype(table.dtype, jnp.integer):
                return perturbed_int_table_lookup(
                    table,
                    indices,
                    _factor_key(slot, self.group),
                    slot.rank,
                    slot.sigma_shift,
                    factor_sign=slot.factor_sign[...],
                )
            return perturbed_table_lookup(
                table, indices, _factor_key(slot, self.group), slot.rank, slot.sigma
            )
        a, b = factors
        if jnp.issubdtype(table.dtype, jnp.integer):
            return perturbed_int_table_lookup_prepared(
                table,
                indices,
                a,
                b,
                slot.sigma_shift,
                factor_sign=slot.factor_sign[...],
            )
        return perturbed_table_lookup_prepared(
            table,
            indices,
            a,
            b,
            slot.sigma,
        )

    def attend(self, query: Array) -> Array:
        """Tied-logit projection using the same table factors as ``__call__``."""
        table = self.embedding[...]
        slot = self.slot
        if slot.enabled:
            if jnp.issubdtype(table.dtype, jnp.integer):
                # Chunked CE should prefer gather+int_matmul; this path is for
                # small-V debugging (row-sparse factors over all ids).
                from ._integer import int_matmul

                idx = jnp.arange(table.shape[0], dtype=jnp.int32)
                e = perturbed_int_table_lookup(
                    table,
                    idx,
                    _factor_key(slot, self.group),
                    slot.rank,
                    slot.sigma_shift,
                    factor_sign=slot.factor_sign[...],
                )
                return int_matmul(query, e.T)
            return perturbed_tied_logits(
                query, table, _factor_key(slot, self.group), slot.rank, slot.sigma
            )
        if jnp.issubdtype(table.dtype, jnp.integer):
            from ._integer import int_matmul

            return int_matmul(query, table.T)
        return query @ table.T


class ZgLayerNorm(nnx.Module):
    """``nnx.LayerNorm`` replacement with factor-only scale/bias perturbations."""

    def __init__(
        self,
        ln: nnx.LayerNorm,
        slot: ZeroGradSlot,
        scale_group: str | None,
        bias_group: str | None,
    ) -> None:
        self.scale = ln.scale
        self.bias = ln.bias
        self.epsilon = ln.epsilon
        self.use_scale = ln.use_scale
        self.use_bias = ln.use_bias
        self.slot = slot
        self.scale_group = scale_group
        self.bias_group = bias_group

    def __call__(self, x: Array) -> Array:
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        y = (x - mean) / jnp.sqrt(var + self.epsilon)
        slot = self.slot
        if self.use_scale and self.scale is not None:
            scale = self.scale[...]
            if slot.enabled and self.scale_group is not None:
                scale = perturbed_vector(
                    scale,
                    _factor_key(slot, self.scale_group),
                    slot.sigma,
                    factor_sign=slot.factor_sign[...],
                )
            y = y * scale
        if self.use_bias and self.bias is not None:
            bias = self.bias[...]
            if slot.enabled and self.bias_group is not None:
                bias = perturbed_vector(
                    bias,
                    _factor_key(slot, self.bias_group),
                    slot.sigma,
                    factor_sign=slot.factor_sign[...],
                )
            y = y + bias
        return y


class ZgVector(nnx.Module):
    """Bare 1-D param wrapper; use as an array (``__jax_array__``) or call ``()``."""

    def __init__(self, param: nnx.Param, slot: ZeroGradSlot, group: str) -> None:
        self.param = param
        self.slot = slot
        self.group = group

    def _read(self) -> Array:
        value = self.param[...]
        slot = self.slot
        if slot.enabled:
            # Perturbation dtype follows the leaf, not the global integer_es flag
            # (ternary kernels are int; γ / RMS / bias stay float).
            if jnp.issubdtype(value.dtype, jnp.integer):
                return perturbed_int_vector(
                    value,
                    _factor_key(slot, self.group),
                    slot.sigma_shift,
                    factor_sign=slot.factor_sign[...],
                )
            return perturbed_vector(
                value,
                _factor_key(slot, self.group),
                slot.sigma,
                factor_sign=slot.factor_sign[...],
            )
        return value

    def __call__(self) -> Array:
        return self._read()

    def __jax_array__(self) -> Array:
        return self._read()


class ZgTable(nnx.Module):
    """Bare 2-D TABLE param wrapper; index with ``table[indices]``."""

    def __init__(self, param: nnx.Param, slot: ZeroGradSlot, group: str) -> None:
        self.param = param
        self.slot = slot
        self.group = group

    def __getitem__(self, indices: Array) -> Array:
        table = self.param[...]
        slot = self.slot
        if slot.enabled:
            if jnp.issubdtype(table.dtype, jnp.integer):
                return perturbed_int_table_lookup(
                    table,
                    indices,
                    _factor_key(slot, self.group),
                    slot.rank,
                    slot.sigma_shift,
                    factor_sign=slot.factor_sign[...],
                )
            return perturbed_table_lookup(
                table, indices, _factor_key(slot, self.group), slot.rank, slot.sigma
            )
        return table[indices]

    def attend(self, query: Array) -> Array:
        table = self.param[...]
        slot = self.slot
        if slot.enabled:
            return perturbed_tied_logits(
                query, table, _factor_key(slot, self.group), slot.rank, slot.sigma
            )
        return query @ table.T


def _is_surged(module: object) -> bool:
    return isinstance(
        module,
        (
            ZgLinear,
            ZgConv,
            ZgIntConv,
            ZgIntLinear,
            ZgIntLUT,
            ZgIntSpatialProj,
            ZgTernaryLinear,
            ZgEmbed,
            ZgLayerNorm,
            ZgVector,
            ZgTable,
            ZeroGradSlot,
        ),
    )


def apply_surgery(
    model: nnx.Module,
    *,
    rank: int,
    sigma: float,
    manifest_version: int = 1,
    sigma_shift: int = 4,
    integer_es: bool = False,
) -> tuple[nnx.Module, Manifest]:
    """Replace stock NNX layers with factor-aware ones and build a Manifest.

    Mutates ``model`` in place (and returns it) so the caller's reference is the
    surgically modified module.
    """
    if not isinstance(model, nnx.Module):
        raise TypeError("apply_surgery expects an nnx.Module")

    if hasattr(model, "zg_slot") and isinstance(getattr(model, "zg_slot"), ZeroGradSlot):
        slot: ZeroGradSlot = model.zg_slot
        slot.rank = rank
        slot.sigma = sigma
        slot.sigma_shift = int(sigma_shift)
        slot.integer_es = bool(integer_es)
    else:
        slot = ZeroGradSlot(
            rank=rank, sigma=sigma, sigma_shift=sigma_shift, integer_es=integer_es
        )
        model.zg_slot = slot

    entries: list[ManifestEntry] = []

    # Pass 1: replace Linear / IntLinear / Embed / LayerNorm (collect entries as we go).
    for path, module in list(nnx.iter_graph(model)):
        if not isinstance(module, nnx.Module) or _is_surged(module):
            continue
        for name, value in list(vars(module).items()):
            child_path = path + (name,)
            if isinstance(value, nnx.Linear):
                leaf = _path_tuple(child_path)
                kg = _path_str(child_path + ("kernel",))
                entries.append(
                    ManifestEntry(leaf + ("kernel",), ParameterLayout.MATRIX, kg)
                )
                bg: str | None = None
                if value.use_bias and value.bias is not None:
                    bg = _path_str(child_path + ("bias",))
                    entries.append(
                        ManifestEntry(leaf + ("bias",), ParameterLayout.VECTOR, bg)
                    )
                setattr(module, name, ZgLinear(value, slot, kg, bg))
            elif isinstance(value, nnx.Conv):
                leaf = _path_tuple(child_path)
                kg = _path_str(child_path + ("kernel",))
                entries.append(
                    ManifestEntry(leaf + ("kernel",), ParameterLayout.MATRIX, kg)
                )
                bg = None
                if value.use_bias and value.bias is not None:
                    bg = _path_str(child_path + ("bias",))
                    entries.append(
                        ManifestEntry(leaf + ("bias",), ParameterLayout.VECTOR, bg)
                    )
                setattr(module, name, ZgConv(value, slot, kg, bg))
            elif isinstance(value, IntLinear):
                leaf = _path_tuple(child_path)
                kg = _path_str(child_path + ("kernel",))
                entries.append(
                    ManifestEntry(leaf + ("kernel",), ParameterLayout.MATRIX, kg)
                )
                bg = None
                if value.use_bias and value.bias is not None:
                    bg = _path_str(child_path + ("bias",))
                    entries.append(
                        ManifestEntry(leaf + ("bias",), ParameterLayout.VECTOR, bg)
                    )
                setattr(module, name, ZgIntLinear(value, slot, kg, bg))
            elif isinstance(value, IntConv):
                leaf = _path_tuple(child_path)
                kg = _path_str(child_path + ("kernel",))
                entries.append(
                    ManifestEntry(leaf + ("kernel",), ParameterLayout.MATRIX, kg)
                )
                bg = None
                if value.use_bias and value.bias is not None:
                    bg = _path_str(child_path + ("bias",))
                    entries.append(
                        ManifestEntry(leaf + ("bias",), ParameterLayout.VECTOR, bg)
                    )
                setattr(module, name, ZgIntConv(value, slot, kg, bg))
            elif isinstance(value, IntLUT):
                leaf = _path_tuple(child_path)
                g = _path_str(child_path + ("table",))
                entries.append(
                    ManifestEntry(leaf + ("table",), ParameterLayout.VECTOR, g)
                )
                setattr(module, name, ZgIntLUT(value, slot, g))
            elif isinstance(value, TernaryLinear):
                leaf = _path_tuple(child_path)
                kg = _path_str(child_path + ("kernel",))
                gg = _path_str(child_path + ("gamma",))
                entries.append(
                    ManifestEntry(leaf + ("kernel",), ParameterLayout.MATRIX, kg)
                )
                entries.append(
                    ManifestEntry(leaf + ("gamma",), ParameterLayout.VECTOR, gg)
                )
                rg: str | None = None
                if value.use_rms_norm and value.rms_scale is not None:
                    rg = _path_str(child_path + ("rms_scale",))
                    entries.append(
                        ManifestEntry(leaf + ("rms_scale",), ParameterLayout.VECTOR, rg)
                    )
                bg = None
                if value.use_bias and value.bias is not None:
                    bg = _path_str(child_path + ("bias",))
                    entries.append(
                        ManifestEntry(leaf + ("bias",), ParameterLayout.VECTOR, bg)
                    )
                setattr(module, name, ZgTernaryLinear(value, slot, kg, gg, rg, bg))
            elif isinstance(value, IntSpatialProj):
                leaf = _path_tuple(child_path)
                kg = _path_str(child_path + ("kernel",))
                bg = _path_str(child_path + ("bias",))
                entries.append(
                    ManifestEntry(leaf + ("kernel",), ParameterLayout.MATRIX, kg)
                )
                entries.append(
                    ManifestEntry(leaf + ("bias",), ParameterLayout.VECTOR, bg)
                )
                setattr(module, name, ZgIntSpatialProj(value, slot, kg, bg))
            elif isinstance(value, nnx.Embed):
                leaf = _path_tuple(child_path)
                g = _path_str(child_path + ("embedding",))
                entries.append(
                    ManifestEntry(leaf + ("embedding",), ParameterLayout.TABLE, g)
                )
                setattr(module, name, ZgEmbed(value, slot, g))
            elif isinstance(value, nnx.LayerNorm):
                leaf = _path_tuple(child_path)
                sg: str | None = None
                bg = None
                if value.use_scale and value.scale is not None:
                    sg = _path_str(child_path + ("scale",))
                    entries.append(
                        ManifestEntry(leaf + ("scale",), ParameterLayout.VECTOR, sg)
                    )
                if value.use_bias and value.bias is not None:
                    bg = _path_str(child_path + ("bias",))
                    entries.append(
                        ManifestEntry(leaf + ("bias",), ParameterLayout.VECTOR, bg)
                    )
                setattr(module, name, ZgLayerNorm(value, slot, sg, bg))

    # Pass 2: wrap bare Params not already owned by surged modules.
    for path, module in list(nnx.iter_graph(model)):
        if not isinstance(module, nnx.Module) or _is_surged(module):
            continue
        for name, value in list(vars(module).items()):
            if not isinstance(value, nnx.Param):
                continue
            child_path = path + (name,)
            # After wrapping, param lives at child_path + ("param",)
            layout = _layout_from_param(value)
            leaf = _path_tuple(child_path)
            if layout is ParameterLayout.VECTOR:
                g = _path_str(child_path + ("param",))
                entries.append(
                    ManifestEntry(leaf + ("param",), ParameterLayout.VECTOR, g)
                )
                setattr(module, name, ZgVector(value, slot, g))
            elif layout is ParameterLayout.TABLE:
                g = _path_str(child_path + ("param",))
                entries.append(
                    ManifestEntry(leaf + ("param",), ParameterLayout.TABLE, g)
                )
                setattr(module, name, ZgTable(value, slot, g))
            elif layout is ParameterLayout.MATRIX:
                raise ValueError(
                    f"Bare 2-D Param at {'.'.join(str(p) for p in child_path)} defaults to "
                    f"MATRIX, which requires nnx.Linear for factor-only matmul. "
                    f"Use nnx.Linear, or mark_table(...) for embedding-style tables."
                )
            else:
                raise ValueError(f"unsupported layout {layout}")

    if not entries:
        raise ValueError("surgery found no trainable parameters to optimize")

    # Stable order by path for deterministic group_key splits.
    entries.sort(key=lambda e: e.path)
    manifest = Manifest(version=manifest_version, entries=tuple(entries))
    slot.manifest = manifest
    slot.rank = rank
    slot.sigma = sigma
    slot.sigma_shift = int(sigma_shift)
    slot.integer_es = bool(integer_es)
    return model, manifest


def _stringify_keys(tree: Any) -> Any:
    """Recursively convert dict keys to strings for Manifest / Optax trees."""
    if isinstance(tree, dict):
        return {str(k): _stringify_keys(v) for k, v in tree.items()}
    return tree


def _restore_keys(tree: Any, template: Any) -> Any:
    """Restore key types (e.g. int List indices) from a template pure dict."""
    if isinstance(tree, dict) and isinstance(template, dict):
        out: dict = {}
        # Map stringified template keys back to original key objects
        tmpl_by_str = {str(k): k for k in template}
        for k, v in tree.items():
            orig_k = tmpl_by_str.get(str(k), k)
            tmpl_v = template.get(orig_k, template.get(k))
            out[orig_k] = _restore_keys(v, tmpl_v) if isinstance(tmpl_v, dict) else v
        return out
    return tree


def params_pure_dict(model: nnx.Module) -> dict:
    """Return a plain nested dict of ``nnx.Param`` arrays for Optax / replay.

    Dict keys are stringified so they match Manifest paths (``nnx.List`` uses
    integer indices in the raw NNX state).
    """
    return _stringify_keys(nnx.state(model, nnx.Param).to_pure_dict())


def update_params(model: nnx.Module, pure_dict: dict) -> None:
    """Write an Optax-updated pure param dict back onto ``model``."""
    template = nnx.state(model, nnx.Param).to_pure_dict()
    nnx.update(model, _restore_keys(pure_dict, template))


def _set_slot_binding(tree: Any, key: Array, factor_sign: Array) -> Any:
    """Return a pure-dict copy with ZeroGradSlot ``key`` / ``factor_sign`` replaced."""
    if not isinstance(tree, dict):
        return tree
    out: dict = {}
    for k, v in tree.items():
        if k in ("slot", "zg_slot") and isinstance(v, dict) and "key" in v:
            slot = {}
            for sk, sv in v.items():
                if sk == "key":
                    slot[sk] = key
                elif sk == "factor_sign":
                    slot[sk] = factor_sign
                else:
                    slot[sk] = _set_slot_binding(sv, key, factor_sign)
            out[k] = slot
        else:
            out[k] = _set_slot_binding(v, key, factor_sign)
    return out


def bind_candidate(
    graphdef: Any,
    state: Any,
    candidate_key_val: Array,
    *,
    enabled: bool = True,
    factor_sign: Array | int = 1,
) -> nnx.Module:
    """Merge graph state with ``candidate_key`` / antithetical sign on the shared slot."""
    sign = jnp.asarray(factor_sign, dtype=jnp.int32)
    # State values must be replaced before entering the candidate's trace
    # context; mutating a merged Variable under vmap raises TraceContextError.
    # Clone only the two slot Variables in flattened state and merge once.
    # This avoids full state -> dict recursion plus an extra merge/split/merge
    # cycle in every candidate trace.
    flat_state = []
    for path, variable in state.flat_state():
        if len(path) >= 2 and path[-2] in ("slot", "zg_slot"):
            if path[-1] == "key":
                variable = variable.replace(candidate_key_val)
            elif path[-1] == "factor_sign":
                variable = variable.replace(sign)
        flat_state.append((path, variable))
    bound_state = type(state).from_flat_path(flat_state)
    model = nnx.merge(graphdef, bound_state)
    if hasattr(model, "zg_slot") and isinstance(model.zg_slot, ZeroGradSlot):
        model.zg_slot.enabled = enabled
    return model


def _set_slot_keys(tree: Any, key: Array) -> Any:
    """Backward-compatible key-only slot rewrite."""
    return _set_slot_binding(tree, key, jnp.int32(1))


def disable_candidates(model: nnx.Module) -> None:
    """Turn off candidate perturbations (eval / inference)."""
    if hasattr(model, "zg_slot") and isinstance(model.zg_slot, ZeroGradSlot):
        model.zg_slot.enabled = False
