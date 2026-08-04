"""Low-rank factor and vector-noise generation for ZeroGrad layouts."""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real

import jax
import jax.numpy as jnp

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


def matrix_factors(key: Array, shape: Sequence[int], rank: int, *, dtype: jnp.dtype) -> tuple[Array, Array]:
    """Draw A[in, rank], B[rank, out] for a matrix leaf shaped [in, out]."""
    key_a, key_b, in_features, out_features = _matrix_leaf_keys(key, shape, rank)
    # Draw in float32 so factor values are stable across forward/replay
    # regardless of the weight dtype a loss_fn may cast to (see issue #17:
    # jax.random.normal produces different values per dtype). Cast to the
    # compute dtype only after the draw, at the use site. Integer weight
    # dtypes keep factors in float32 (cast-to-int8 would destroy the signal).
    out_dtype = factor_compute_dtype(dtype)
    return (
        jax.random.normal(key_a, (in_features, rank), dtype=jnp.float32).astype(out_dtype),
        jax.random.normal(key_b, (rank, out_features), dtype=jnp.float32).astype(out_dtype),
    )


def int_matrix_factors(key: Array, shape: Sequence[int], rank: int) -> tuple[Array, Array]:
    """Appendix H.1: int8 factors ``round(16·N(0,1))`` for a matrix leaf ``[in, out]``."""
    from ._eggroll_h import int8_from_normal

    key_a, key_b, in_features, out_features = _matrix_leaf_keys(key, shape, rank)
    return (
        int8_from_normal(key_a, (in_features, rank)),
        int8_from_normal(key_b, (rank, out_features)),
    )


def stacked_matrix_factors(
    key: Array,
    shape: Sequence[int],
    rank: int,
    *,
    dtype: jnp.dtype,
) -> tuple[Array, Array]:
    """Draw independent matrix factors for ``[layers, in, out]`` weights."""
    layers, in_features, out_features = _validate_shape(shape, 3, "stacked matrix shape")
    layer_ids = jnp.arange(layers, dtype=jnp.int32)

    def one(layer_id):
        return matrix_factors(
            jax.random.fold_in(key, layer_id),
            (in_features, out_features),
            rank,
            dtype=dtype,
        )

    return jax.vmap(one)(layer_ids)


def stacked_int_matrix_factors(
    key: Array,
    shape: Sequence[int],
    rank: int,
) -> tuple[Array, Array]:
    """Draw independent int8 factors for ``[layers, in, out]`` weights."""
    layers, in_features, out_features = _validate_shape(shape, 3, "stacked matrix shape")
    layer_ids = jnp.arange(layers, dtype=jnp.int32)

    def one(layer_id):
        return int_matrix_factors(
            jax.random.fold_in(key, layer_id),
            (in_features, out_features),
            rank,
        )

    return jax.vmap(one)(layer_ids)


def _table_leaf_keys(
    key: Array, shape: Sequence[int], rank: int
) -> tuple[Array, Array, int, int]:
    """Validate table shape/rank and split the PRNG key for A/B factors."""
    rows, columns = _validate_shape(shape, 2, "table shape")
    _validate_rank(rank)
    key_a, key_b = jax.random.split(key)
    return key_a, key_b, rows, columns


def table_factors(key: Array, shape: Sequence[int], rank: int, *, dtype: jnp.dtype) -> tuple[Array, Array]:
    """Draw A[rows, rank], B[cols, rank] for a table leaf shaped [rows, cols]."""
    key_a, key_b, rows, columns = _table_leaf_keys(key, shape, rank)
    # See matrix_factors: draw in float32 for dtype-stable parity, then cast.
    out_dtype = factor_compute_dtype(dtype)
    return (
        jax.random.normal(key_a, (rows, rank), dtype=jnp.float32).astype(out_dtype),
        jax.random.normal(key_b, (columns, rank), dtype=jnp.float32).astype(out_dtype),
    )


def int_table_factors(key: Array, shape: Sequence[int], rank: int) -> tuple[Array, Array]:
    """Appendix H.1: int8 table factors for a leaf shaped ``[rows, cols]``.

    Uses one bulk draw for ``A[rows, rank]`` (Qwen-scale ``V·r`` is ~0.5MB).
    Per-row ``fold_in`` would make ES updates O(V) RNG and dominate step time.
    """
    from ._eggroll_h import int8_from_normal

    key_a, key_b, rows, columns = _table_leaf_keys(key, shape, rank)
    return (
        int8_from_normal(key_a, (rows, rank)),
        int8_from_normal(key_b, (columns, rank)),
    )


def vector_noise(key: Array, shape: Sequence[int], *, dtype: jnp.dtype) -> Array:
    """Draw IID standard-normal perturbation for a one-dimensional leaf."""
    (size,) = _validate_shape(shape, 1, "vector shape")
    # See matrix_factors: draw in float32 for dtype-stable parity, then cast.
    out_dtype = factor_compute_dtype(dtype)
    return jax.random.normal(key, (size,), dtype=jnp.float32).astype(out_dtype)


def int_vector_noise(key: Array, shape: Sequence[int]) -> Array:
    """Appendix H.1: int8 IID noise for a one-dimensional leaf."""
    from ._eggroll_h import int8_from_normal

    (size,) = _validate_shape(shape, 1, "vector shape")
    return int8_from_normal(key, (size,))


def stacked_vector_noise(
    key: Array,
    shape: Sequence[int],
    *,
    dtype: jnp.dtype,
) -> Array:
    """Draw independent vector noise for ``[layers, size]`` parameters."""
    layers, size = _validate_shape(shape, 2, "stacked vector shape")
    return jax.vmap(
        lambda layer_id: vector_noise(
            jax.random.fold_in(key, layer_id), (size,), dtype=dtype
        )
    )(jnp.arange(layers, dtype=jnp.int32))


def stacked_int_vector_noise(key: Array, shape: Sequence[int]) -> Array:
    """Draw independent int8 noise for ``[layers, size]`` parameters."""
    layers, size = _validate_shape(shape, 2, "stacked vector shape")
    return jax.vmap(
        lambda layer_id: int_vector_noise(
            jax.random.fold_in(key, layer_id), (size,)
        )
    )(jnp.arange(layers, dtype=jnp.int32))
