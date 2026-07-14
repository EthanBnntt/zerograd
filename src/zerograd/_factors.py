"""Low-rank factor and vector-noise generation for ZeroGrad layouts."""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real

import jax
import jax.numpy as jnp

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
    return jnp.asarray(sigma / math.sqrt(rank), dtype=dtype)


def matrix_factors(key: Array, shape: Sequence[int], rank: int, *, dtype: jnp.dtype) -> tuple[Array, Array]:
    """Draw A[in, rank], B[rank, out] for a matrix leaf shaped [in, out]."""
    in_features, out_features = _validate_shape(shape, 2, "matrix shape")
    _validate_rank(rank)
    key_a, key_b = jax.random.split(key)
    # Draw in float32 so factor values are stable across forward/replay
    # regardless of the weight dtype a loss_fn may cast to (see issue #17:
    # jax.random.normal produces different values per dtype). Cast to the
    # requested dtype only after the draw, at the use site.
    return (
        jax.random.normal(key_a, (in_features, rank), dtype=jnp.float32).astype(dtype),
        jax.random.normal(key_b, (rank, out_features), dtype=jnp.float32).astype(dtype),
    )


def table_factors(key: Array, shape: Sequence[int], rank: int, *, dtype: jnp.dtype) -> tuple[Array, Array]:
    """Draw A[rows, rank], B[cols, rank] for a table leaf shaped [rows, cols]."""
    rows, columns = _validate_shape(shape, 2, "table shape")
    _validate_rank(rank)
    key_a, key_b = jax.random.split(key)
    # See matrix_factors: draw in float32 for dtype-stable parity, then cast.
    return (
        jax.random.normal(key_a, (rows, rank), dtype=jnp.float32).astype(dtype),
        jax.random.normal(key_b, (columns, rank), dtype=jnp.float32).astype(dtype),
    )


def vector_noise(key: Array, shape: Sequence[int], *, dtype: jnp.dtype) -> Array:
    """Draw IID standard-normal perturbation for a one-dimensional leaf."""
    (size,) = _validate_shape(shape, 1, "vector shape")
    # See matrix_factors: draw in float32 for dtype-stable parity, then cast.
    return jax.random.normal(key, (size,), dtype=jnp.float32).astype(dtype)
