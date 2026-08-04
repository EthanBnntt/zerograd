"""Factor-only forward operations for one deterministic ZeroGrad candidate."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from ._factors import matrix_factors, scaled_factor, table_factors, vector_noise
from ._integer import factor_compute_dtype
from ._keys import group_key
from ._manifest import Manifest, ParameterLayout, ParameterPath, ParameterTree

Array = jax.Array


def perturbed_linear(
    x: Array,
    weight: Array,
    key: Array,
    rank: int,
    sigma: float,
    *,
    factor_sign: Array | int = 1,
) -> Array:
    """Evaluate ``x @ weight + factor_sign * sigma * (x @ A) @ B / sqrt(rank)``."""
    if weight.ndim != 2 or x.shape[-1] != weight.shape[0]:
        raise ValueError("linear input and [in, out] weight shapes are incompatible")
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
    from ._factors import int_matrix_factors

    if weight.ndim != 2 or x.shape[-1] != weight.shape[0]:
        raise ValueError("linear input and [in, out] weight shapes are incompatible")
    if not isinstance(sigma_shift, int) or isinstance(sigma_shift, bool) or sigma_shift < 0:
        raise ValueError(f"sigma_shift must be a non-negative int, got {sigma_shift!r}")
    from ._integer import int_matmul

    # Keep int8 operands so XLA can lower to WMMA IU8 (not int32×int32).
    y = int_matmul(x, weight)
    a, b = int_matrix_factors(key, weight.shape, rank)
    a = (a.astype(jnp.int32) * jnp.asarray(factor_sign, dtype=jnp.int32)).astype(jnp.int8)
    # Int8×int8 → int32 factor path; right-shift replaces float σ/√r.
    pert = int_matmul(int_matmul(x, a), b)
    shift = 4 + int(sigma_shift)
    return y + (pert >> shift)


def perturbed_conv(
    x: Array,
    weight_2d: Array,
    key: Array,
    rank: int,
    sigma: float,
    *,
    kernel_hw: tuple[int, int],
    in_features: int,
    strides: tuple[int, ...],
    padding,
    lhs_dilation: tuple[int, ...],
    rhs_dilation: tuple[int, ...],
    feature_group_count: int = 1,
    precision=None,
    factor_sign: Array | int = 1,
    conv_general_dilated=None,
) -> Array:
    """Factor-only NHWC convolution: ``conv(x,W) + s·σ/√r · conv(x,A) @ B``.

    ``weight_2d`` is the flattened kernel ``[kH·kW·in, out]`` (MATRIX layout).
    """
    if weight_2d.ndim != 2:
        raise ValueError("conv weights must be stored as a 2-D MATRIX leaf")
    kh, kw = kernel_hw
    fan_in = kh * kw * int(in_features)
    if weight_2d.shape[0] != fan_in:
        raise ValueError(
            f"conv weight fan-in {weight_2d.shape[0]} != {kh}*{kw}*{in_features}"
        )
    out_features = weight_2d.shape[1]
    a, b = matrix_factors(key, weight_2d.shape, rank, dtype=weight_2d.dtype)
    sign = jnp.asarray(factor_sign, dtype=factor_compute_dtype(weight_2d.dtype))
    scale = sign * scaled_factor(rank, sigma, weight_2d.dtype)

    from flax.nnx.nn.linear import _conv_dimension_numbers

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
    padding: str | tuple = "SAME",
    factor_sign: Array | int = 1,
) -> Array:
    """Factor-only int8 NHWC conv via im2col + :func:`perturbed_int_linear`.

    Returns the **int32** accumulator (caller requantizes). ``weight_2d`` is the
    flattened MATRIX leaf ``[kH·kW·in, out]``.
    """
    if weight_2d.ndim != 2:
        raise ValueError("int conv weights must be stored as a 2-D MATRIX leaf")
    if not isinstance(sigma_shift, int) or isinstance(sigma_shift, bool) or sigma_shift < 0:
        raise ValueError(f"sigma_shift must be a non-negative int, got {sigma_shift!r}")
    kh, kw = kernel_hw
    fan_in = kh * kw * int(in_features)
    if weight_2d.shape[0] != fan_in:
        raise ValueError(
            f"int conv weight fan-in {weight_2d.shape[0]} != {kh}*{kw}*{in_features}"
        )
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
    y = perturbed_int_linear(
        flat, weight_2d, key, rank, sigma_shift, factor_sign=factor_sign
    )
    return y.reshape(bsz, h, w, int(weight_2d.shape[1]))


def perturbed_int_vector(
    vector: Array,
    key: Array,
    sigma_shift: int,
    *,
    factor_sign: Array | int = 1,
) -> Array:
    """Appendix H-style int8 vector noise with right-shift scale."""
    from ._factors import int_vector_noise

    if vector.ndim != 1:
        raise ValueError("vector weights must be one-dimensional")
    if not isinstance(sigma_shift, int) or isinstance(sigma_shift, bool) or sigma_shift < 0:
        raise ValueError(f"sigma_shift must be a non-negative int, got {sigma_shift!r}")
    noise = int_vector_noise(key, vector.shape).astype(jnp.int32) * jnp.asarray(
        factor_sign, dtype=jnp.int32
    )
    shift = 4 + int(sigma_shift)
    # Toward-zero scale (``>>`` and Python ``//`` both bias negatives toward -1).
    delta = jax.lax.div(noise, jnp.int32(1 << shift))
    info = jnp.iinfo(vector.dtype)
    return jnp.clip(vector.astype(jnp.int32) + delta, info.min, info.max).astype(vector.dtype)


def perturbed_int_table_lookup(
    table: Array,
    indices: Array,
    key: Array,
    rank: int,
    sigma_shift: int,
    *,
    factor_sign: Array | int = 1,
) -> Array:
    """Appendix H.1 table gather: ``table[i] + ((A[i] @ B) >> (4+σ̂))`` in int.

    Draws bulk ``A[V,r]`` (small at Qwen scale) then gathers rows — same factors
    as :func:`zerograd._replay.replay_entry_integer` for ES correctness.
    """
    from ._factors import int_table_factors

    if table.ndim != 2:
        raise ValueError("table weights must be two-dimensional")
    a, b = int_table_factors(key, table.shape, rank)
    return perturbed_int_table_lookup_prepared(
        table,
        indices,
        a,
        b,
        sigma_shift,
        factor_sign=factor_sign,
    )


def perturbed_int_table_lookup_prepared(
    table: Array,
    indices: Array,
    a: Array,
    b: Array,
    sigma_shift: int,
    *,
    factor_sign: Array | int = 1,
) -> Array:
    """Gather an integer table using factors generated once for the candidate.

    This is algebraically identical to :func:`perturbed_int_table_lookup` but
    lets callers reuse ``A[rows, rank]`` and ``B[cols, rank]`` across input
    embedding, target lookup, and every chunk of a tied vocabulary head.
    """
    if table.ndim != 2:
        raise ValueError("table weights must be two-dimensional")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("prepared table factors must be two-dimensional")
    if a.shape[0] != table.shape[0] or b.shape[0] != table.shape[1]:
        raise ValueError("prepared table factors are incompatible with table shape")
    if a.shape[1] != b.shape[1]:
        raise ValueError("prepared table factors must share the same rank")
    if not isinstance(sigma_shift, int) or isinstance(sigma_shift, bool) or sigma_shift < 0:
        raise ValueError(f"sigma_shift must be a non-negative int, got {sigma_shift!r}")
    sign = jnp.asarray(factor_sign, dtype=jnp.int32)
    a = (a.astype(jnp.int32) * sign).astype(jnp.int8)
    base = table[indices].astype(jnp.int32)
    pert = jnp.einsum(
        "...r,cr->...c", a[indices].astype(jnp.int32), b.astype(jnp.int32)
    )
    shift = 4 + int(sigma_shift)
    info = jnp.iinfo(table.dtype)
    return jnp.clip(base + (pert >> shift), info.min, info.max).astype(table.dtype)


def perturbed_table_lookup(table: Array, indices: Array, key: Array, rank: int, sigma: float) -> Array:
    """Gather table rows with factor-only table perturbation."""
    if table.ndim != 2:
        raise ValueError("table weights must be two-dimensional")
    a, b = table_factors(key, table.shape, rank, dtype=table.dtype)
    scale = scaled_factor(rank, sigma, table.dtype)
    if jnp.issubdtype(table.dtype, jnp.integer):
        # Gather int rows; add rounded float perturbation in int32 then clip.
        base = table[indices].astype(jnp.int32)
        pert = scale * jnp.einsum("...r,cr->...c", a[indices], b)
        info = jnp.iinfo(table.dtype)
        return jnp.clip(base + jnp.rint(pert).astype(jnp.int32), info.min, info.max).astype(
            table.dtype
        )
    return table[indices] + scale * jnp.einsum("...r,cr->...c", a[indices], b)


def perturbed_table_lookup_prepared(
    table: Array,
    indices: Array,
    a: Array,
    b: Array,
    sigma: float,
) -> Array:
    """Float/integer table lookup with reusable pre-generated factors."""
    if table.ndim != 2 or a.ndim != 2 or b.ndim != 2:
        raise ValueError("table and prepared factors must be two-dimensional")
    rank = int(a.shape[1])
    if b.shape != (table.shape[1], rank) or a.shape[0] != table.shape[0]:
        raise ValueError("prepared table factors are incompatible with table shape")
    scale = scaled_factor(rank, sigma, table.dtype)
    if jnp.issubdtype(table.dtype, jnp.integer):
        base = table[indices].astype(jnp.int32)
        pert = scale * jnp.einsum("...r,cr->...c", a[indices], b)
        info = jnp.iinfo(table.dtype)
        return jnp.clip(
            base + jnp.rint(pert).astype(jnp.int32), info.min, info.max
        ).astype(table.dtype)
    return table[indices] + scale * jnp.einsum("...r,cr->...c", a[indices], b)


def perturbed_tied_logits(x: Array, table: Array, key: Array, rank: int, sigma: float) -> Array:
    """Project with a table and the same factor algebra used by table lookup."""
    if table.ndim != 2 or x.shape[-1] != table.shape[1]:
        raise ValueError("logit input and [rows, cols] table shapes are incompatible")
    a, b = table_factors(key, table.shape, rank, dtype=table.dtype)
    scale = scaled_factor(rank, sigma, table.dtype)
    if jnp.issubdtype(table.dtype, jnp.integer):
        from ._integer import int_matmul

        y = int_matmul(x, table.T)
        pert = scale * jnp.einsum("...c,cr,vr->...v", x.astype(jnp.float32), b, a)
        return y + jnp.rint(pert).astype(jnp.int32)
    return x @ table.T + scale * jnp.einsum("...c,cr,vr->...v", x, b, a)


def perturbed_vector(
    vector: Array,
    key: Array,
    sigma: float,
    *,
    factor_sign: Array | int = 1,
) -> Array:
    """Return a vector leaf plus deterministic IID normal noise."""
    if vector.ndim != 1:
        raise ValueError("vector weights must be one-dimensional")
    noise = vector_noise(key, vector.shape, dtype=vector.dtype)
    scale = scaled_factor(1, sigma, vector.dtype)
    sign = jnp.asarray(factor_sign, dtype=jnp.float32)
    if jnp.issubdtype(vector.dtype, jnp.integer):
        delta = jnp.rint(sign * scale * noise).astype(jnp.int32)
        info = jnp.iinfo(vector.dtype)
        return jnp.clip(vector.astype(jnp.int32) + delta, info.min, info.max).astype(vector.dtype)
    return vector + (sign * scale).astype(vector.dtype) * noise.astype(vector.dtype)


@dataclass(frozen=True, slots=True)
class CandidateContext:
    """Candidate-local manifest lookup and factor-only forward-operation facade."""

    manifest: Manifest
    candidate_key: Array
    rank: int
    sigma: float

    def key_for(self, path: ParameterPath) -> Array:
        """Return the deterministic group key for one manifest path."""
        return group_key(self.candidate_key, self.manifest, self.manifest.entry(path).group)

    def linear(self, params: ParameterTree, path: ParameterPath, x: Array) -> Array:
        """Evaluate a manifest-selected matrix leaf without materializing a delta."""
        entry = self.manifest.entry(path)
        if entry.layout is not ParameterLayout.MATRIX:
            raise ValueError(f"{'.'.join(path)} is not a matrix entry")
        return perturbed_linear(x, self.manifest.resolve(params, path), self.key_for(path), self.rank, self.sigma)

    def table_lookup(self, params: ParameterTree, path: ParameterPath, indices: Array) -> Array:
        """Look up rows from a manifest-selected table without a dense delta."""
        entry = self.manifest.entry(path)
        if entry.layout is not ParameterLayout.TABLE:
            raise ValueError(f"{'.'.join(path)} is not a table entry")
        return perturbed_table_lookup(self.manifest.resolve(params, path), indices, self.key_for(path), self.rank, self.sigma)

    def tied_logits(self, params: ParameterTree, path: ParameterPath, x: Array) -> Array:
        """Project through a manifest-selected table with its shared factors."""
        entry = self.manifest.entry(path)
        if entry.layout is not ParameterLayout.TABLE:
            raise ValueError(f"{'.'.join(path)} is not a table entry")
        return perturbed_tied_logits(x, self.manifest.resolve(params, path), self.key_for(path), self.rank, self.sigma)

    def vector(self, params: ParameterTree, path: ParameterPath) -> Array:
        """Return a manifest-selected vector with deterministic IID noise."""
        entry = self.manifest.entry(path)
        if entry.layout is not ParameterLayout.VECTOR:
            raise ValueError(f"{'.'.join(path)} is not a vector entry")
        return perturbed_vector(self.manifest.resolve(params, path), self.key_for(path), self.sigma)
