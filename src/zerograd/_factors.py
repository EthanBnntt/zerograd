"""Low-rank factor and vector-noise generation for ZeroGrad layouts."""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real

import jax
import jax.numpy as jnp

from ._eggroll_h import int8_from_normal
from ._integer import factor_compute_dtype

Array = jax.Array


def _validate_rank(rank: int) -> None:
    if not isinstance(rank, int) or isinstance(rank, bool) or rank < 1:
        raise ValueError(f"rank must be a positive integer, got {rank!r}")


def _validate_shape(shape: Sequence[int], ndim: int, name: str) -> tuple[int, ...]:
    dimensions = tuple(shape)
    if len(dimensions) != ndim or any(
        not isinstance(size, int) or isinstance(size, bool) or size < 1
        for size in dimensions
    ):
        raise ValueError(f"{name} must be a {ndim}-D shape of positive integers, got {shape!r}")
    return dimensions


def scaled_factor(rank: int, sigma: float, dtype: jnp.dtype) -> Array:
    """Return the shared perturbation scale ``sigma / sqrt(rank)``."""
    _validate_rank(rank)
    if (
        not isinstance(sigma, Real)
        or isinstance(sigma, bool)
        or not math.isfinite(sigma)
        or sigma <= 0
    ):
        raise ValueError(f"sigma must be a finite positive real, got {sigma!r}")
    # Integer leaves keep the scale in float32; casting sigma/√r to int8 is useless.
    out_dtype = factor_compute_dtype(dtype)
    return jnp.asarray(sigma / math.sqrt(rank), dtype=out_dtype)


def _matrix_leaf_keys(
    key: Array, shape: Sequence[int], rank: int
) -> tuple[Array, Array, int, int]:
    """Validate matrix shape/rank and split the PRNG key for A/B factors."""
    in_features, out_features = _validate_shape(shape, 2, "matrix shape")
    _validate_rank(rank)
    key_a, key_b = jax.random.split(key)
    return key_a, key_b, in_features, out_features


def _table_leaf_keys(
    key: Array, shape: Sequence[int], rank: int
) -> tuple[Array, Array, int, int]:
    """Validate table shape/rank and split the PRNG key for A/B factors."""
    rows, columns = _validate_shape(shape, 2, "table shape")
    _validate_rank(rank)
    key_a, key_b = jax.random.split(key)
    return key_a, key_b, rows, columns


def _draw_normal_pair(
    key_a: Array,
    key_b: Array,
    shape_a: tuple[int, ...],
    shape_b: tuple[int, ...],
    *,
    dtype: jnp.dtype | None,
    integer: bool,
) -> tuple[Array, Array]:
    """Draw A/B factors: Appendix H int8, or float32→compute-dtype normals."""
    if integer:
        return int8_from_normal(key_a, shape_a), int8_from_normal(key_b, shape_b)
    assert dtype is not None
    # Draw in float32 so factor values are stable across forward/replay
    # regardless of the weight dtype a loss_fn may cast to (see issue #17:
    # jax.random.normal produces different values per dtype). Cast to the
    # compute dtype only after the draw, at the use site. Integer weight
    # dtypes keep factors in float32 (cast-to-int8 would destroy the signal).
    out_dtype = factor_compute_dtype(dtype)
    return (
        jax.random.normal(key_a, shape_a, dtype=jnp.float32).astype(out_dtype),
        jax.random.normal(key_b, shape_b, dtype=jnp.float32).astype(out_dtype),
    )


def _draw_normal_vector(
    key: Array,
    shape: tuple[int, ...],
    *,
    dtype: jnp.dtype | None,
    integer: bool,
) -> Array:
    if integer:
        return int8_from_normal(key, shape)
    assert dtype is not None
    out_dtype = factor_compute_dtype(dtype)
    return jax.random.normal(key, shape, dtype=jnp.float32).astype(out_dtype)


def _stacked_vmap(key: Array, layers: int, fn):
    """Vmap ``fn(fold_in(key, layer_id))`` over ``layers``."""
    return jax.vmap(lambda layer_id: fn(jax.random.fold_in(key, layer_id)))(
        jnp.arange(layers, dtype=jnp.int32)
    )


def matrix_factors(
    key: Array,
    shape: Sequence[int],
    rank: int,
    *,
    dtype: jnp.dtype | None = None,
    integer: bool = False,
) -> tuple[Array, Array]:
    """Draw A[in, rank], B[rank, out] for a matrix leaf shaped [in, out].

    Float path (``integer=False``) requires ``dtype`` and draws normals in
    float32 then casts to the compute dtype. Integer path (``integer=True``)
    draws Appendix H int8 ``round(16·N(0,1))`` factors.
    """
    if not integer and dtype is None:
        raise TypeError("matrix_factors requires dtype=... unless integer=True")
    key_a, key_b, in_features, out_features = _matrix_leaf_keys(key, shape, rank)
    return _draw_normal_pair(
        key_a,
        key_b,
        (in_features, rank),
        (rank, out_features),
        dtype=dtype,
        integer=integer,
    )


def int_matrix_factors(key: Array, shape: Sequence[int], rank: int) -> tuple[Array, Array]:
    """Appendix H.1: int8 factors ``round(16·N(0,1))`` for a matrix leaf ``[in, out]``."""
    return matrix_factors(key, shape, rank, integer=True)


def stacked_matrix_factors(
    key: Array,
    shape: Sequence[int],
    rank: int,
    *,
    dtype: jnp.dtype | None = None,
    integer: bool = False,
) -> tuple[Array, Array]:
    """Draw independent matrix factors for ``[layers, in, out]`` weights."""
    if not integer and dtype is None:
        raise TypeError("stacked_matrix_factors requires dtype=... unless integer=True")
    layers, in_features, out_features = _validate_shape(shape, 3, "stacked matrix shape")
    return _stacked_vmap(
        key,
        layers,
        lambda k: matrix_factors(
            k, (in_features, out_features), rank, dtype=dtype, integer=integer
        ),
    )


def stacked_int_matrix_factors(
    key: Array,
    shape: Sequence[int],
    rank: int,
) -> tuple[Array, Array]:
    """Draw independent int8 factors for ``[layers, in, out]`` weights."""
    return stacked_matrix_factors(key, shape, rank, integer=True)


def table_factors(
    key: Array,
    shape: Sequence[int],
    rank: int,
    *,
    dtype: jnp.dtype | None = None,
    integer: bool = False,
) -> tuple[Array, Array]:
    """Draw A[rows, rank], B[cols, rank] for a table leaf shaped [rows, cols].

    Integer path uses one bulk draw for ``A[rows, rank]`` (Qwen-scale ``V·r`` is
    ~0.5MB). Per-row ``fold_in`` would make ES updates O(V) RNG and dominate
    step time.
    """
    if not integer and dtype is None:
        raise TypeError("table_factors requires dtype=... unless integer=True")
    key_a, key_b, rows, columns = _table_leaf_keys(key, shape, rank)
    return _draw_normal_pair(
        key_a,
        key_b,
        (rows, rank),
        (columns, rank),
        dtype=dtype,
        integer=integer,
    )


def int_table_factors(key: Array, shape: Sequence[int], rank: int) -> tuple[Array, Array]:
    """Appendix H.1: int8 table factors for a leaf shaped ``[rows, cols]``."""
    return table_factors(key, shape, rank, integer=True)


def vector_noise(
    key: Array,
    shape: Sequence[int],
    *,
    dtype: jnp.dtype | None = None,
    integer: bool = False,
) -> Array:
    """Draw IID perturbation for a one-dimensional leaf (float normal or int8)."""
    if not integer and dtype is None:
        raise TypeError("vector_noise requires dtype=... unless integer=True")
    (size,) = _validate_shape(shape, 1, "vector shape")
    return _draw_normal_vector(key, (size,), dtype=dtype, integer=integer)


def int_vector_noise(key: Array, shape: Sequence[int]) -> Array:
    """Appendix H.1: int8 IID noise for a one-dimensional leaf."""
    return vector_noise(key, shape, integer=True)


def stacked_vector_noise(
    key: Array,
    shape: Sequence[int],
    *,
    dtype: jnp.dtype | None = None,
    integer: bool = False,
) -> Array:
    """Draw independent vector noise for ``[layers, size]`` parameters."""
    if not integer and dtype is None:
        raise TypeError("stacked_vector_noise requires dtype=... unless integer=True")
    layers, size = _validate_shape(shape, 2, "stacked vector shape")
    return _stacked_vmap(
        key,
        layers,
        lambda k: vector_noise(k, (size,), dtype=dtype, integer=integer),
    )


def stacked_int_vector_noise(key: Array, shape: Sequence[int]) -> Array:
    """Draw independent int8 noise for ``[layers, size]`` parameters."""
    return stacked_vector_noise(key, shape, integer=True)
