"""Discrete integer bin updates from accumulated perturbation evidence."""

from __future__ import annotations

import math
from typing import Any, cast

import jax
import jax.numpy as jnp

from ._factors import INT8_FACTOR_SCALE
from ._manifest import Manifest, ParameterLayout, ParameterTree, ThresholdTree

Array = jax.Array


def update_alpha_schedule(generation: int, *, base: float = 1.0, decay: float = 0.015) -> float:
    """Decay the fraction of integer leaves updated each generation.

    ``α_t = base / (decay * t + 1)``. Algorithm from Eggroll (Appendix H.3).
    """
    if generation < 0:
        raise ValueError(f"generation must be non-negative, got {generation}")
    if not math.isfinite(base) or base <= 0 or base > 1:
        raise ValueError(f"base alpha must be in (0, 1], got {base!r}")
    if not math.isfinite(decay) or decay < 0:
        raise ValueError(f"decay must be a non-negative float, got {decay!r}")
    return float(base / (decay * generation + 1.0))


def _normal_ppf(p: float) -> float:
    """Inverse CDF of standard normal: ``Φ⁻¹(p)``."""
    erfinv = getattr(math, "erfinv", None)
    if erfinv is None:
        # Winitzki approximation for erfinv
        x = 2.0 * p - 1.0
        a = 0.147
        ln = math.log(1.0 - x * x)
        term = 2.0 / (math.pi * a) + ln / 2.0
        z_erf = math.copysign(
            math.sqrt(math.sqrt(term * term - ln / a) - term),
            x,
        )
        return math.sqrt(2.0) * z_erf
    return math.sqrt(2.0) * erfinv(2.0 * p - 1.0)


def bin_update_threshold(
    alpha: float,
    num_directions: int,
    *,
    layout: ParameterLayout | str = ParameterLayout.MATRIX,
    rank: int = 1,
) -> int:
    """Layout-scaled evidence threshold for a single ±1 bin move.

    Null evidence scale differs by factor algebra (Eggroll Appendix H.3):

    - **MATRIX / TABLE** — ``E = Σ_p Σ_r F·(A_r B_r)`` with two int8 factors →
      ``⌊16 · z · 16 · √(N·rank)⌋``. Rank enters because evidence sums ``rank``
      independent factor products per direction; omitting it makes the bar too
      low for ``rank>1`` and the realized flip rate grows with rank.
    - **VECTOR** — ``E = Σ F·ε`` with one int8 factor (no rank axis) →
      ``⌊16 · z · √N⌋``.

    Using the matrix threshold for vectors makes ``|E_vec|`` never clear the bar
    (LUTs / biases stay frozen). ``num_directions`` is ``N/2`` with antithetical pairs.
    """
    if not math.isfinite(alpha) or not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
    if num_directions < 1:
        raise ValueError(f"num_directions must be >= 1, got {num_directions}")
    if not isinstance(rank, int) or isinstance(rank, bool) or rank < 1:
        raise ValueError(f"rank must be a positive integer, got {rank!r}")
    layout = ParameterLayout(layout)
    z = _normal_ppf(1.0 - alpha / 2.0)
    # Common: z · √N · 16¹
    scale = INT8_FACTOR_SCALE * z * math.sqrt(float(num_directions))
    if layout in (ParameterLayout.VECTOR, ParameterLayout.STACKED_VECTOR):
        return math.floor(scale)
    # MATRIX / TABLE: second factor of 16 from A@B / table factors, and √rank
    # from summing rank independent products into each evidence entry.
    return math.floor(scale * INT8_FACTOR_SCALE * math.sqrt(float(rank)))


def threshold_tree_for_manifest(
    params: ParameterTree,
    manifest: Manifest,
    *,
    alpha: float,
    num_directions: int,
    rank: int = 1,
) -> ThresholdTree:
    """Build a threshold pytree aligned with ``params`` (layout-scaled per leaf)."""
    thr_by_path = {
        entry.path: bin_update_threshold(
            alpha, num_directions, layout=entry.layout, rank=rank
        )
        for entry in manifest.entries
    }

    def _build(node: ParameterTree | Array, path: tuple[str, ...]) -> Any:
        if isinstance(node, dict):
            return {k: _build(v, (*path, k)) for k, v in node.items()}
        if path in thr_by_path:
            return thr_by_path[path]
        # Non-manifest leaf (should not be updated): impossible threshold.
        return 2**30

    return cast(ThresholdTree, _build(params, ()))


def apply_bin_updates(
    params: Array | ParameterTree,
    evidence: Array | ParameterTree,
    threshold: int | Array | ThresholdTree,
    *,
    qmin: int | None = None,
    qmax: int | None = None,
) -> Array | ParameterTree:
    """Move each integer leaf by one discrete bin when ``|E| > threshold``.

    ``threshold`` may be a scalar (same bar for every leaf) or a pytree matching
    ``params`` (from :func:`threshold_tree_for_manifest`).

    Optional ``qmin``/``qmax`` override the dtype iinfo range (e.g. int4 in int8 storage).
    """
    if isinstance(params, dict) and isinstance(evidence, dict):
        out: dict = {}
        if isinstance(threshold, dict):
            for k, v in params.items():
                e = evidence.get(k)
                if e is None:
                    out[k] = v
                else:
                    out[k] = apply_bin_updates(v, e, threshold[k], qmin=qmin, qmax=qmax)
        else:
            for k, v in params.items():
                e = evidence.get(k)
                if e is None:
                    out[k] = v
                else:
                    out[k] = apply_bin_updates(v, e, threshold, qmin=qmin, qmax=qmax)
        return out
    if isinstance(params, jax.Array) and isinstance(evidence, jax.Array):
        if not jnp.issubdtype(params.dtype, jnp.integer):
            return params
        if isinstance(threshold, dict):
            raise TypeError("threshold tree does not align with parameter arrays")
        info = jnp.iinfo(params.dtype)
        # Optional int4-in-int8 clip only applies to int8 leaves; wider ints keep iinfo.
        if qmin is not None and qmax is not None and params.dtype == jnp.int8:
            lo, hi = int(qmin), int(qmax)
        else:
            lo, hi = int(info.min), int(info.max)
        step = jnp.sign(evidence.astype(jnp.int32))
        threshold_i32 = jnp.asarray(threshold, dtype=jnp.int32)
        mask = (jnp.abs(evidence.astype(jnp.int32)) > threshold_i32).astype(jnp.int32)
        delta = step * mask
        return jnp.clip(params.astype(jnp.int32) + delta, lo, hi).astype(params.dtype)
    return params


def apply_bin_updates_jit(
    params: ParameterTree,
    evidence: ParameterTree,
    thresholds: ThresholdTree,
) -> ParameterTree:
    """Bin-update an entire param tree as one fused XLA program (H100-friendly).

    Equivalent to :func:`apply_bin_updates`; thresholds are folded as static
    Python ints (not traced) so the jitted function only takes array args.
    """

    def _update(params_p, evidence_p):
        return apply_bin_updates(params_p, evidence_p, thresholds)

    return jax.jit(_update)(params, evidence)
