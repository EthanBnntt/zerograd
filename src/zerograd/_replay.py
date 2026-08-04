"""Factor replay: reconstruct parameter-space pseudo-gradients from the same keys used in forward evaluation."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ._factors import (
    int_matrix_factors,
    int_table_factors,
    int_vector_noise,
    matrix_factors,
    scaled_factor,
    stacked_int_matrix_factors,
    stacked_int_vector_noise,
    stacked_matrix_factors,
    stacked_vector_noise,
    table_factors,
    vector_noise,
)
from ._keys import candidate_key, group_key
from ._manifest import Manifest, ParameterLayout, ParameterPath, ParameterTree

Array = jax.Array

# Tile big int tables on device during replay (Qwen embeds) to bound peak
# [tile, cols] int32 evidence; each tile is still one bulk RNG + GEMM.
_TABLE_ROW_TILE = 65536


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
            if parameter.ndim == 3:
                return stacked_matrix_factors(
                    gk, parameter.shape, rank, dtype=parameter.dtype
                )
            return matrix_factors(gk, parameter.shape, rank, dtype=parameter.dtype)
        elif entry.layout is ParameterLayout.TABLE:
            return table_factors(gk, parameter.shape, rank, dtype=parameter.dtype)
        else:
            if parameter.ndim == 2:
                return stacked_vector_noise(
                    gk, parameter.shape, dtype=parameter.dtype
                )
            return vector_noise(gk, parameter.shape, dtype=parameter.dtype)

    if entry.layout is ParameterLayout.MATRIX:
        a_pop, b_pop = jax.vmap(factors_for_candidate)(candidate_ids)
        weight_shape = (shaped_weights.shape[0],) + (1,) * (a_pop.ndim - 1)
        weighted_a = a_pop * shaped_weights.reshape(weight_shape)
        return jnp.einsum(
            "p...ir,p...ro->...io", weighted_a, b_pop
        ) * scaled_factor(rank, 1.0, parameter.dtype)
    elif entry.layout is ParameterLayout.TABLE:
        a_pop, b_pop = jax.vmap(factors_for_candidate)(candidate_ids)
        weighted_a = a_pop * shaped_weights[:, None, None]
        return jnp.einsum("pxr,pyr->xy", weighted_a, b_pop) * scaled_factor(rank, 1.0, parameter.dtype)
    else:
        noise_pop = jax.vmap(factors_for_candidate)(candidate_ids)
        weight_shape = (shaped_weights.shape[0],) + (1,) * (noise_pop.ndim - 1)
        return jnp.sum(
            noise_pop * shaped_weights.reshape(weight_shape), axis=0
        ) * scaled_factor(1, 1.0, parameter.dtype)


def replay_entry_integer(
    params: ParameterTree,
    manifest: Manifest,
    path: ParameterPath,
    base_key: Array,
    pair_ids: Array,
    shaped_weights: Array,
    rank: int,
) -> Array:
    """Appendix H.3: int32 evidence ``E = Σ_i F_i · (A_i B_i)`` from int8 factors.

    ``pair_ids`` indexes the ``+`` half of antithetical pairs (same keys as forward).
    ``shaped_weights`` are int ``{-1,0,1}`` from :func:`shape_antithetical_loss`.
    """
    entry = manifest.entry(path)
    parameter = manifest.resolve(params, path)
    group = entry.group
    # Float leaves (γ, RMS, bias, embeddings) use float factors even under
    # integer_es; only integer kernels use Appendix H int8 factors.
    param_is_float = jnp.issubdtype(parameter.dtype, jnp.floating)

    def factors_for_pair(pid):
        ck = candidate_key(base_key, pid)
        gk = group_key(ck, manifest, group)
        if param_is_float:
            if entry.layout is ParameterLayout.MATRIX:
                if parameter.ndim == 3:
                    return stacked_matrix_factors(
                        gk, parameter.shape, rank, dtype=parameter.dtype
                    )
                return matrix_factors(gk, parameter.shape, rank, dtype=parameter.dtype)
            if entry.layout is ParameterLayout.TABLE:
                return table_factors(gk, parameter.shape, rank, dtype=parameter.dtype)
            if parameter.ndim == 2:
                return stacked_vector_noise(
                    gk, parameter.shape, dtype=parameter.dtype
                )
            return vector_noise(gk, parameter.shape, dtype=parameter.dtype)
        if entry.layout is ParameterLayout.MATRIX:
            if parameter.ndim == 3:
                return stacked_int_matrix_factors(gk, parameter.shape, rank)
            return int_matrix_factors(gk, parameter.shape, rank)
        if entry.layout is ParameterLayout.TABLE:
            return int_table_factors(gk, parameter.shape, rank)
        if parameter.ndim == 2:
            return stacked_int_vector_noise(gk, parameter.shape)
        return int_vector_noise(gk, parameter.shape)

    f = shaped_weights
    # Allow int {-1,0,1} (bin path) or float centered weights (Adam path).
    use_float = param_is_float or jnp.issubdtype(
        getattr(f, "dtype", jnp.float32), jnp.floating
    )
    if entry.layout is ParameterLayout.MATRIX:
        a_pop, b_pop = jax.vmap(factors_for_pair)(pair_ids)
        weight_shape = (f.shape[0],) + (1,) * (a_pop.ndim - 1)
        if use_float:
            weighted_a = a_pop.astype(jnp.float32) * f.astype(jnp.float32).reshape(
                weight_shape
            )
            return jnp.einsum(
                "p...ir,p...ro->...io", weighted_a, b_pop.astype(jnp.float32)
            )
        weighted_a = a_pop.astype(jnp.int32) * f.astype(jnp.int32).reshape(
            weight_shape
        )
        return jnp.einsum(
            "p...ir,p...ro->...io", weighted_a, b_pop.astype(jnp.int32)
        )
    if entry.layout is ParameterLayout.TABLE:
        if not use_float:
            return _int_table_evidence(
                parameter, base_key, group, pair_ids, f, rank, manifest
            )
        a_pop, b_pop = jax.vmap(factors_for_pair)(pair_ids)
        weighted_a = a_pop.astype(jnp.float32) * f.astype(jnp.float32)[:, None, None]
        return jnp.einsum("pxr,pyr->xy", weighted_a, b_pop.astype(jnp.float32))
    noise_pop = jax.vmap(factors_for_pair)(pair_ids)
    weight_shape = (f.shape[0],) + (1,) * (noise_pop.ndim - 1)
    if use_float:
        return jnp.sum(
            noise_pop.astype(jnp.float32)
            * f.astype(jnp.float32).reshape(weight_shape),
            axis=0,
        )
    return jnp.sum(
        noise_pop.astype(jnp.int32) * f.astype(jnp.int32).reshape(weight_shape),
        axis=0,
    )


def _int_table_evidence(
    parameter: Array,
    base_key: Array,
    group: str,
    pair_ids: Array,
    shaped_weights: Array,
    rank: int,
    manifest: Manifest,
) -> Array:
    """Int table evidence with on-device row tiling (Qwen-scale embeds).

    Computes a bulk ``A``/``B`` table once, then for each row-tile does one
    ``Σ_p w_p A_p[rows] @ B_pᵀ`` GEMM and ``lax.dynamic_update_slice`` into a
    preallocated ``[V, cols]`` int32 output. This avoids a big per-tile concat
    while keeping all work on GPU (no Python per-row loop).
    """
    rows, cols = int(parameter.shape[0]), int(parameter.shape[1])
    tile = _TABLE_ROW_TILE if rows > _TABLE_ROW_TILE else rows
    n_tiles = (rows + tile - 1) // tile

    def factors_for_pair(pid):
        ck = candidate_key(base_key, pid)
        gk = group_key(ck, manifest, group)
        return int_table_factors(gk, parameter.shape, rank)

    a_pop, b_pop = jax.vmap(factors_for_pair)(pair_ids)
    w = shaped_weights.astype(jnp.int32)
    out0 = jnp.zeros((rows, cols), dtype=jnp.int32)

    def one_tile(carry, i):
        start = i * tile
        rows_i = jnp.arange(tile, dtype=jnp.int32) + start
        rows_i = jnp.minimum(rows_i, rows - 1)
        a_t = a_pop[:, rows_i, :]
        aw = a_t.astype(jnp.int32) * w[:, None, None]
        ev = jnp.einsum("pxr,pyr->xy", aw, b_pop.astype(jnp.int32))
        valid = (rows_i < rows).astype(ev.dtype)[:, None]
        ev = ev * valid
        return jax.lax.dynamic_update_slice(carry, ev, (start, 0)), ()

    ev, _ = jax.lax.scan(one_tile, out0, jnp.arange(n_tiles, dtype=jnp.int32))
    return ev


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


def replay_integer(
    params: ParameterTree,
    manifest: Manifest,
    base_key: Array,
    pair_ids: Array,
    shaped_weights: Array,
    rank: int,
) -> dict[str, "jax.Array | dict"]:
    """Appendix H.3 int32 evidence tree for discrete bin updates."""
    result: dict[str, "jax.Array | dict"] = {}
    for entry in manifest.entries:
        leaf = replay_entry_integer(
            params, manifest, entry.path, base_key, pair_ids, shaped_weights, rank
        )
        _insert_nested(result, entry.path, leaf)
    return result


def replay_integer_tiled(
    params: ParameterTree,
    manifest: Manifest,
    base_key: Array,
    pair_ids: Array,
    shaped_weights: Array,
    rank: int,
) -> dict[str, "jax.Array | dict"]:
    """Alias for :func:`replay_integer` (int table leaves are internally tiled)."""
    return replay_integer(params, manifest, base_key, pair_ids, shaped_weights, rank)


def _insert_nested(tree: dict, path: tuple[str, ...], value: "jax.Array") -> None:
    """Insert a value at a nested tuple path into a dict tree."""
    node = tree
    for part in path[:-1]:
        if part not in node:
            node[part] = {}
        node = node[part]
    node[path[-1]] = value
