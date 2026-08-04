"""Eggroll Appendix H helpers: int8 factors, antithetical shaping, discrete bin updates."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from ._manifest import Manifest, ParameterLayout, ParameterTree

Array = jax.Array

# Factors are round(16 * N(0,1)) clipped to int8 (Appendix H.1 / G.3).
INT8_FACTOR_SCALE = 16


def int8_from_normal(key: Array, shape: tuple[int, ...]) -> Array:
    """Sample Appendix H int8 noise: ``round(16 * N(0,1))`` clipped to ``[-128, 127]``."""
    raw = jax.random.normal(key, shape, dtype=jnp.float32) * float(INT8_FACTOR_SCALE)
    return jnp.clip(jnp.rint(raw), -128, 127).astype(jnp.int8)


def antithetical_pair_id(candidate_id: Array, population_size: int) -> Array:
    """Map a candidate id to its shared antithetical pair index in ``[0, N/2)``."""
    half = population_size // 2
    return jnp.asarray(candidate_id, dtype=jnp.int32) % half


def antithetical_sign(candidate_id: Array, population_size: int) -> Array:
    """``+1`` for the first half of the population, ``-1`` for the antithetical half."""
    half = population_size // 2
    return jnp.where(jnp.asarray(candidate_id, dtype=jnp.int32) < half, jnp.int32(1), jnp.int32(-1))


def shape_antithetical_loss(losses: Array) -> Array:
    """Appendix H.2: per-pair weights ``sign(L_- - L_+)`` in ``{-1, 0, 1}``.

    Population must be even. Index ``i`` (``0..N/2-1``) is the ``+`` perturbation;
    ``i + N/2`` is the antithetical ``-`` perturbation. Positive weight means the
    ``+`` direction had lower loss and should be reinforced.
    """
    if losses.ndim != 1:
        raise ValueError(f"losses must be one-dimensional, got shape {losses.shape}")
    n = int(losses.shape[0])
    if n < 2 or n % 2 != 0:
        raise ValueError(f"antithetical shaping requires even population_size >= 2, got {n}")
    half = n // 2
    loss_pos = losses[:half]
    loss_neg = losses[half:]
    # Lower loss on + side → reinforce + factors.
    return jnp.sign(loss_neg - loss_pos).astype(jnp.int32)


def update_alpha_schedule(generation: int, *, base: float = 1.0, decay: float = 0.015) -> float:
    """Appendix H.3: ``α_t = base / (decay * t + 1)`` (fraction of params updated)."""
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
) -> int:
    """Appendix H.3 layout-scaled bin threshold.

    Null evidence scale differs by factor algebra:

    - **MATRIX / TABLE** — ``E = Σ F·(A B)`` with two int8 factors →
      ``⌊16 · z · 16 · √N⌋`` (Eggroll default).
    - **VECTOR** — ``E = Σ F·ε`` with one int8 factor →
      ``⌊16 · z · √N⌋``.

    Using the matrix threshold for vectors makes ``|E_vec|`` never clear the bar
    (LUTs / biases stay frozen). ``num_directions`` is ``N/2`` with antithetical pairs.
    """
    if not math.isfinite(alpha) or not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
    if num_directions < 1:
        raise ValueError(f"num_directions must be >= 1, got {num_directions}")
    layout = ParameterLayout(layout)
    z = _normal_ppf(1.0 - alpha / 2.0)
    # Common: z · √N · 16¹
    scale = INT8_FACTOR_SCALE * z * math.sqrt(float(num_directions))
    if layout is ParameterLayout.VECTOR:
        return int(math.floor(scale))
    # MATRIX / TABLE: second factor of 16 from A@B / table factors.
    return int(math.floor(scale * INT8_FACTOR_SCALE))


def threshold_tree_for_manifest(
    params: ParameterTree,
    manifest: Manifest,
    *,
    alpha: float,
    num_directions: int,
) -> ParameterTree:
    """Build a threshold pytree aligned with ``params`` (layout-scaled per leaf)."""
    thr_by_path = {
        entry.path: bin_update_threshold(alpha, num_directions, layout=entry.layout)
        for entry in manifest.entries
    }

    def _build(node: ParameterTree | Array, path: tuple[str, ...]) -> ParameterTree | int:
        if isinstance(node, dict):
            return {k: _build(v, path + (k,)) for k, v in node.items()}
        if path in thr_by_path:
            return thr_by_path[path]
        # Non-manifest leaf (should not be updated): impossible threshold.
        return 2**30

    return _build(params, ())  # type: ignore[return-value]


def apply_bin_updates(
    params: Array | dict,
    evidence: Array | dict,
    threshold: int | Array | dict,
    *,
    qmin: int | None = None,
    qmax: int | None = None,
) -> Array | dict:
    """Move each integer leaf by one discrete bin when ``|E| > threshold``.

    ``threshold`` may be a scalar (same bar for every leaf) or a pytree matching
    ``params`` (from :func:`threshold_tree_for_manifest`). Use
    :func:`apply_bin_updates_jit` for a single device program over big trees.

    Optional ``qmin``/``qmax`` override the dtype iinfo range (e.g. int4 in int8 storage).
    """
    if isinstance(params, dict) and isinstance(evidence, dict):
        out: dict = {}
        thr_is_tree = isinstance(threshold, dict)
        for k, v in params.items():
            e = evidence.get(k)
            if e is None:
                out[k] = v
            else:
                thr_k = threshold[k] if thr_is_tree else threshold  # type: ignore[index]
                out[k] = apply_bin_updates(v, e, thr_k, qmin=qmin, qmax=qmax)
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
    thresholds: ParameterTree,
) -> ParameterTree:
    """Bin-update an entire param tree as one fused XLA program (H100-friendly).

    Equivalent to :func:`apply_bin_updates`; thresholds are folded as static
    Python ints (not traced) so the jitted function only takes array args.
    """
    def _update(params_p, evidence_p):
        return apply_bin_updates(params_p, evidence_p, thresholds)
    return jax.jit(_update)(params, evidence)
