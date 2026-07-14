"""Factor replay: reconstruct parameter-space pseudo-gradients from the same keys used in forward evaluation."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ._factors import matrix_factors, scaled_factor, table_factors, vector_noise
from ._keys import candidate_key, group_key
from ._manifest import Manifest, ParameterLayout, ParameterPath, ParameterTree

Array = jax.Array


def replay_entry(
    params: ParameterTree,
    manifest: Manifest,
    path: ParameterPath,
    base_key: Array,
    candidate_ids: Array,
    shaped_weights: Array,
    rank: int,
) -> Array:
    """Replay one manifest entry's pseudo-gradient descent contribution.

    Regenerates the same factors used in forward evaluation and contracts them
    with shaped weights via einsum, never materializing a dense
    ``[population, *parameter_shape]`` tensor for matrix/table layouts.
    """
    entry = manifest.entry(path)
    parameter = manifest.resolve(params, path)
    group = entry.group

    def factors_for_candidate(cid):
        ck = candidate_key(base_key, cid)
        gk = group_key(ck, manifest, group)
        if entry.layout is ParameterLayout.MATRIX:
            return matrix_factors(gk, parameter.shape, rank, dtype=parameter.dtype)
        elif entry.layout is ParameterLayout.TABLE:
            return table_factors(gk, parameter.shape, rank, dtype=parameter.dtype)
        else:
            return vector_noise(gk, parameter.shape, dtype=parameter.dtype)

    if entry.layout is ParameterLayout.MATRIX:
        a_pop, b_pop = jax.vmap(factors_for_candidate)(candidate_ids)
        weighted_a = a_pop * shaped_weights[:, None, None]
        return jnp.einsum("pir,pro->io", weighted_a, b_pop) * scaled_factor(rank, 1.0, parameter.dtype)
    elif entry.layout is ParameterLayout.TABLE:
        a_pop, b_pop = jax.vmap(factors_for_candidate)(candidate_ids)
        weighted_a = a_pop * shaped_weights[:, None, None]
        return jnp.einsum("pxr,pyr->xy", weighted_a, b_pop) * scaled_factor(rank, 1.0, parameter.dtype)
    else:
        noise_pop = jax.vmap(factors_for_candidate)(candidate_ids)
        return jnp.sum(noise_pop * shaped_weights[:, None], axis=0) * scaled_factor(1, 1.0, parameter.dtype)


def replay(
    params: ParameterTree,
    manifest: Manifest,
    base_key: Array,
    candidate_ids: Array,
    shaped_weights: Array,
    rank: int,
) -> dict[str, "jax.Array | dict"]:
    """Reconstruct the full parameter-space descent direction for all manifest entries."""
    result: dict[str, "jax.Array | dict"] = {}
    for entry in manifest.entries:
        leaf = replay_entry(params, manifest, entry.path, base_key, candidate_ids, shaped_weights, rank)
        _insert_nested(result, entry.path, leaf)
    return result


def _insert_nested(tree: dict, path: tuple[str, ...], value: "jax.Array") -> None:
    """Insert a value at a nested tuple path into a dict tree."""
    node = tree
    for part in path[:-1]:
        if part not in node:
            node[part] = {}
        node = node[part]
    node[path[-1]] = value
