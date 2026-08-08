"""Factor-only forward operations for one deterministic ZeroGrad candidate."""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
from flax.nnx.nn.linear import _conv_dimension_numbers

from ._factors import matrix_factors, scaled_factor, table_factors, vector_noise
from ._integer import factor_compute_dtype, int_matmul

Array = jax.Array


def _validate_sigma_shift(sigma_shift: int) -> None:
    if not isinstance(sigma_shift, int) or isinstance(sigma_shift, bool) or sigma_shift < 0:
        raise ValueError(f"sigma_shift must be a non-negative int, got {sigma_shift!r}")


def _validate_linear_shapes(x: Array, weight: Array) -> None:
    if weight.ndim != 2 or x.shape[-1] != weight.shape[0]:
        raise ValueError("linear input and [in, out] weight shapes are incompatible")


def _validate_table_2d(table: Array) -> None:
    if table.ndim != 2:
        raise ValueError("table weights must be two-dimensional")


def _validate_vector_1d(vector: Array) -> None:
    if vector.ndim != 1:
        raise ValueError("vector weights must be one-dimensional")


def _validate_conv_weight_2d(weight_2d: Array, kernel_hw: tuple[int, int], in_features: int) -> tuple[int, int, int]:
    if weight_2d.ndim != 2:
        raise ValueError("conv weights must be stored as a 2-D MATRIX leaf")
    kh, kw = kernel_hw
    fan_in = kh * kw * int(in_features)
    if weight_2d.shape[0] != fan_in:
        raise ValueError(
            f"conv weight fan-in {weight_2d.shape[0]} != {kh}*{kw}*{in_features}"
        )
    return kh, kw, fan_in


def perturbed_linear(
    x: Array,
    weight: Array,
    key: Array,
    rank: int,
    sigma: float | None = None,
    *,
    sigma_shift: int | None = None,
    factor_sign: Array | int = 1,
) -> Array:
    """Evaluate factor-only linear: float ``x@W + s·σ/√r·(x@A)@B`` or int Appendix H.

    Integer weights use ``sigma_shift`` (``σ = 2**(-σ̂)``) and return an int32
    accumulator. Float weights use ``sigma`` and stay in the weight dtype.
    """
    _validate_linear_shapes(x, weight)
    if jnp.issubdtype(weight.dtype, jnp.integer):
        if sigma_shift is None:
            raise ValueError("sigma_shift is required for integer linear weights")
        _validate_sigma_shift(sigma_shift)

        # Keep int8 operands so XLA can lower to WMMA IU8 (not int32×int32).
        y = int_matmul(x, weight)
        a, b = matrix_factors(key, weight.shape, rank, integer=True)
        a = (a.astype(jnp.int32) * jnp.asarray(factor_sign, dtype=jnp.int32)).astype(
            jnp.int8
        )
        # Int8×int8 → int32 factor path; right-shift replaces float σ/√r.
        pert = int_matmul(int_matmul(x, a), b)
        return y + (pert >> (4 + int(sigma_shift)))

    if sigma is None:
        raise ValueError("sigma is required for float linear weights")
    a, b = matrix_factors(key, weight.shape, rank, dtype=weight.dtype)
    sign = jnp.asarray(factor_sign, dtype=factor_compute_dtype(weight.dtype))
    return x @ weight + sign * scaled_factor(rank, sigma, weight.dtype) * ((x @ a) @ b)


def perturbed_int_linear(
    x: Array,
    weight: Array,
    key: Array,
    rank: int,
    sigma_shift: int,
    *,
    factor_sign: Array | int = 1,
) -> Array:
    """Appendix H.1 integer matmul perturbation (int8 factors, int32 accum, shift).

    Returns the **int32** accumulator::

        x @ W + factor_sign * (((x @ A) @ B) >> (4 + sigma_shift))

    with ``A, B`` drawn as ``round(16·N(0,1))`` int8. ``sigma = 2**(-sigma_shift)``.
    """
    return perturbed_linear(
        x, weight, key, rank, sigma_shift=sigma_shift, factor_sign=factor_sign
    )


def perturbed_conv(
    x: Array,
    weight_2d: Array,
    key: Array,
    rank: int,
    sigma: float | None = None,
    *,
    sigma_shift: int | None = None,
    kernel_hw: tuple[int, int],
    in_features: int,
    strides: tuple[int, ...] = (1, 1),
    padding: str | Sequence[tuple[int, int]] = "SAME",
    lhs_dilation: tuple[int, ...] = (1, 1),
    rhs_dilation: tuple[int, ...] = (1, 1),
    feature_group_count: int = 1,
    precision=None,
    factor_sign: Array | int = 1,
    conv_general_dilated=None,
) -> Array:
    """Factor-only NHWC convolution (float dilations, or int via im2col).

    ``weight_2d`` is the flattened kernel ``[kH·kW·in, out]`` (MATRIX layout).
    Integer weights ignore dilation kwargs and route through im2col + linear.
    """
    kh, kw, fan_in = _validate_conv_weight_2d(weight_2d, kernel_hw, in_features)

    if jnp.issubdtype(weight_2d.dtype, jnp.integer):
        if sigma_shift is None:
            raise ValueError("sigma_shift is required for integer conv weights")
        _validate_sigma_shift(sigma_shift)
        # One patch extract, then the same factor-only matmul as IntLinear.
        patches = jax.lax.conv_general_dilated_patches(
            x,
            filter_shape=(kh, kw),
            window_strides=strides,
            padding=padding,
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )
        bsz, h, w, _ = patches.shape
        flat = patches.reshape(bsz * h * w, fan_in)
        y = perturbed_linear(
            flat,
            weight_2d,
            key,
            rank,
            sigma_shift=sigma_shift,
            factor_sign=factor_sign,
        )
        return y.reshape(bsz, h, w, int(weight_2d.shape[1]))

    if sigma is None:
        raise ValueError("sigma is required for float conv weights")
    out_features = weight_2d.shape[1]
    a, b = matrix_factors(key, weight_2d.shape, rank, dtype=weight_2d.dtype)
    sign = jnp.asarray(factor_sign, dtype=factor_compute_dtype(weight_2d.dtype))
    scale = sign * scaled_factor(rank, sigma, weight_2d.dtype)

    if conv_general_dilated is None:
        conv_general_dilated = jax.lax.conv_general_dilated
    dimension_numbers = _conv_dimension_numbers(x.shape)
    w4 = weight_2d.reshape(kh, kw, in_features, out_features)
    a4 = a.reshape(kh, kw, in_features, rank)
    y = conv_general_dilated(
        x,
        w4,
        strides,
        padding,
        lhs_dilation=lhs_dilation,
        rhs_dilation=rhs_dilation,
        dimension_numbers=dimension_numbers,
        feature_group_count=feature_group_count,
        precision=precision,
    )
    yp = conv_general_dilated(
        x,
        a4,
        strides,
        padding,
        lhs_dilation=lhs_dilation,
        rhs_dilation=rhs_dilation,
        dimension_numbers=dimension_numbers,
        feature_group_count=feature_group_count,
        precision=precision,
    )
    return y + scale.astype(y.dtype) * (yp @ b.astype(yp.dtype))


def perturbed_int_conv(
    x: Array,
    weight_2d: Array,
    key: Array,
    rank: int,
    sigma_shift: int,
    *,
    kernel_hw: tuple[int, int],
    in_features: int,
    strides: tuple[int, int] = (1, 1),
    padding: str | Sequence[tuple[int, int]] = "SAME",
    factor_sign: Array | int = 1,
) -> Array:
    """Factor-only int8 NHWC conv via im2col + :func:`perturbed_int_linear`.

    Returns the **int32** accumulator (caller requantizes). ``weight_2d`` is the
    flattened MATRIX leaf ``[kH·kW·in, out]``.
    """
    return perturbed_conv(
        x,
        weight_2d,
        key,
        rank,
        sigma_shift=sigma_shift,
        kernel_hw=kernel_hw,
        in_features=in_features,
        strides=strides,
        padding=padding,
        factor_sign=factor_sign,
    )


def perturbed_vector(
    vector: Array,
    key: Array,
    sigma: float | None = None,
    *,
    sigma_shift: int | None = None,
    factor_sign: Array | int = 1,
) -> Array:
    """Return a vector leaf plus deterministic noise (float normal or int Appendix H)."""
    _validate_vector_1d(vector)
    if jnp.issubdtype(vector.dtype, jnp.integer):
        if sigma_shift is None:
            raise ValueError("sigma_shift is required for integer vector weights")
        _validate_sigma_shift(sigma_shift)
        noise = vector_noise(key, vector.shape, integer=True).astype(jnp.int32) * jnp.asarray(
            factor_sign, dtype=jnp.int32
        )
        shift = 4 + int(sigma_shift)
        # Toward-zero scale (``>>`` and Python ``//`` both bias negatives toward -1).
        delta = jax.lax.div(noise, jnp.int32(1 << shift))
        info = jnp.iinfo(vector.dtype)
        return jnp.clip(vector.astype(jnp.int32) + delta, info.min, info.max).astype(
            vector.dtype
        )

    if sigma is None:
        raise ValueError("sigma is required for float vector weights")
    noise = vector_noise(key, vector.shape, dtype=vector.dtype)
    scale = scaled_factor(1, sigma, vector.dtype)
    sign = jnp.asarray(factor_sign, dtype=jnp.float32)
    return vector + (sign * scale).astype(vector.dtype) * noise.astype(vector.dtype)


def perturbed_int_vector(
    vector: Array,
    key: Array,
    sigma_shift: int,
    *,
    factor_sign: Array | int = 1,
) -> Array:
    """Appendix H-style int8 vector noise with right-shift scale."""
    return perturbed_vector(
        vector, key, sigma_shift=sigma_shift, factor_sign=factor_sign
    )


def perturbed_table_lookup_prepared(
    table: Array,
    indices: Array,
    a: Array,
    b: Array,
    sigma: float | None = None,
    *,
    sigma_shift: int | None = None,
    factor_sign: Array | int = 1,
) -> Array:
    """Table gather with reusable pre-generated factors (float or int)."""
    _validate_table_2d(table)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("prepared table factors must be two-dimensional")
    if jnp.issubdtype(table.dtype, jnp.integer):
        if a.shape[0] != table.shape[0] or b.shape[0] != table.shape[1]:
            raise ValueError("prepared table factors are incompatible with table shape")
        if a.shape[1] != b.shape[1]:
            raise ValueError("prepared table factors must share the same rank")
        if sigma_shift is None:
            raise ValueError("sigma_shift is required for integer table weights")
        _validate_sigma_shift(sigma_shift)
        sign = jnp.asarray(factor_sign, dtype=jnp.int32)
        a = (a.astype(jnp.int32) * sign).astype(jnp.int8)
        base = table[indices].astype(jnp.int32)
        pert = jnp.einsum(
            "...r,cr->...c", a[indices].astype(jnp.int32), b.astype(jnp.int32)
        )
        info = jnp.iinfo(table.dtype)
        return jnp.clip(base + (pert >> (4 + int(sigma_shift))), info.min, info.max).astype(
            table.dtype
        )

    rank = int(a.shape[1])
    if b.shape != (table.shape[1], rank) or a.shape[0] != table.shape[0]:
        raise ValueError("prepared table factors are incompatible with table shape")
    if sigma is None:
        raise ValueError("sigma is required for float table weights")
    scale = scaled_factor(rank, sigma, table.dtype)
    return table[indices] + scale * jnp.einsum("...r,cr->...c", a[indices], b)


def perturbed_table_lookup(
    table: Array,
    indices: Array,
    key: Array,
    rank: int,
    sigma: float | None = None,
    *,
    sigma_shift: int | None = None,
    factor_sign: Array | int = 1,
) -> Array:
    """Gather table rows with factor-only table perturbation (float or int)."""
    _validate_table_2d(table)
    if jnp.issubdtype(table.dtype, jnp.integer):
        a, b = table_factors(key, table.shape, rank, integer=True)
        return perturbed_table_lookup_prepared(
            table,
            indices,
            a,
            b,
            sigma_shift=sigma_shift,
            factor_sign=factor_sign,
        )
    a, b = table_factors(key, table.shape, rank, dtype=table.dtype)
    return perturbed_table_lookup_prepared(table, indices, a, b, sigma=sigma)


def perturbed_int_table_lookup(
    table: Array,
    indices: Array,
    key: Array,
    rank: int,
    sigma_shift: int,
    *,
    factor_sign: Array | int = 1,
) -> Array:
    """Integer table gather; thin wrapper around :func:`perturbed_table_lookup`.

    Requires ``sigma_shift`` (Appendix H.1) instead of float ``sigma``. Row factors
    are drawn via :func:`~zerograd._factors.table_factors` inside
    :func:`perturbed_table_lookup`, then gathered at ``indices`` — same algebra as
    :func:`~zerograd._replay.replay_entry_integer` for ES replay parity.
    """
    return perturbed_table_lookup(
        table,
        indices,
        key,
        rank,
        sigma_shift=sigma_shift,
        factor_sign=factor_sign,
    )


def perturbed_tied_logits(
    x: Array,
    table: Array,
    key: Array,
    rank: int,
    sigma: float,
) -> Array:
    """Project with a float table and the same factor algebra used by table lookup."""
    if table.ndim != 2 or x.shape[-1] != table.shape[1]:
        raise ValueError("logit input and [rows, cols] table shapes are incompatible")
    a, b = table_factors(key, table.shape, rank, dtype=table.dtype)
    scale = scaled_factor(rank, sigma, table.dtype)
    return x @ table.T + scale * jnp.einsum("...c,cr,vr->...v", x, b, a)
