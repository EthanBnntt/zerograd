"""Factor-only forward operations for one deterministic ZeroGrad candidate."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from ._factors import matrix_factors, scaled_factor, table_factors, vector_noise
from ._keys import group_key
from ._manifest import Manifest, ParameterLayout, ParameterPath, ParameterTree

Array = jax.Array


def perturbed_linear(x: Array, weight: Array, key: Array, rank: int, sigma: float) -> Array:
    """Evaluate ``x @ weight + sigma * (x @ A) @ B / sqrt(rank)``."""
    if weight.ndim != 2 or x.shape[-1] != weight.shape[0]:
        raise ValueError("linear input and [in, out] weight shapes are incompatible")
    a, b = matrix_factors(key, weight.shape, rank, dtype=weight.dtype)
    return x @ weight + scaled_factor(rank, sigma, weight.dtype) * ((x @ a) @ b)


def perturbed_table_lookup(table: Array, indices: Array, key: Array, rank: int, sigma: float) -> Array:
    """Gather table rows with factor-only table perturbation."""
    if table.ndim != 2:
        raise ValueError("table weights must be two-dimensional")
    a, b = table_factors(key, table.shape, rank, dtype=table.dtype)
    return table[indices] + scaled_factor(rank, sigma, table.dtype) * jnp.einsum("...r,cr->...c", a[indices], b)


def perturbed_tied_logits(x: Array, table: Array, key: Array, rank: int, sigma: float) -> Array:
    """Project with a table and the same factor algebra used by table lookup."""
    if table.ndim != 2 or x.shape[-1] != table.shape[1]:
        raise ValueError("logit input and [rows, cols] table shapes are incompatible")
    a, b = table_factors(key, table.shape, rank, dtype=table.dtype)
    return x @ table.T + scaled_factor(rank, sigma, table.dtype) * jnp.einsum("...c,cr,vr->...v", x, b, a)


def perturbed_vector(vector: Array, key: Array, sigma: float) -> Array:
    """Return a vector leaf plus deterministic IID normal noise."""
    if vector.ndim != 1:
        raise ValueError("vector weights must be one-dimensional")
    return vector + scaled_factor(1, sigma, vector.dtype) * vector_noise(key, vector.shape, dtype=vector.dtype)


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
