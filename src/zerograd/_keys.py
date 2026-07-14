"""Deterministic PRNG derivation for ZeroGrad replay."""

from __future__ import annotations

import zlib

import jax
import jax.numpy as jnp

from ._manifest import Manifest

Array = jax.Array


def _positive_int(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def step_key(seed: int, run_id: str, generation: int, manifest_version: int) -> Array:
    """Return the stable base key for one logical optimizer generation."""
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run_id must be a non-empty string")
    if not isinstance(generation, int) or isinstance(generation, bool) or generation < 0:
        raise ValueError("generation must be a non-negative integer")
    _positive_int(manifest_version, "manifest_version")
    payload = f"{seed}:{run_id}:{manifest_version}:{generation}".encode()
    return jax.random.key(zlib.crc32(payload) & 0xFFFFFFFF)


def candidate_key(base_key: Array, candidate_id: int | Array) -> Array:
    """Derive one global non-negative candidate key without worker partitioning."""
    if isinstance(candidate_id, bool):
        raise TypeError("candidate_id must be a non-negative integer scalar")
    if isinstance(candidate_id, int):
        if candidate_id < 0:
            raise ValueError("candidate_id must be non-negative")
        return jax.random.fold_in(base_key, candidate_id)
    candidate = jnp.asarray(candidate_id)
    if candidate.ndim != 0 or not jnp.issubdtype(candidate.dtype, jnp.integer):
        raise TypeError("candidate_id must be an integer scalar")
    # Reject negative concrete scalars to match the int branch. Traced
    # scalars (e.g. inside jax.vmap over a non-negative candidate-id array)
    # cannot be concretized here, so the check is skipped in that case —
    # the library only ever vmaps over jnp.arange(0, population).
    try:
        if int(candidate) < 0:
            raise ValueError("candidate_id must be non-negative")
    except jax.errors.ConcretizationTypeError:
        pass
    return jax.random.fold_in(base_key, candidate)


def group_key(candidate: Array, manifest: Manifest, group: str) -> Array:
    """Derive a manifest-order-stable key for one selected parameter group."""
    return jax.random.split(candidate, len(manifest.entries))[manifest.group_index(group)]
