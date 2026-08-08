"""Centered-loss ES fitness shaping for ZeroGrad."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

Array = jax.Array


def shape_centered_loss(losses: Array, sigma: float) -> Array:
    """Return shaped weights ``-(losses - mean(losses)) / (population_size * sigma)``.

    These weights contract candidate perturbations into a loss-DESCENT direction:
    candidates with below-mean loss receive positive weight, pulling parameters
    toward them. ``ZeroGrad`` negates this descent direction before handing it to
    Optax as a conventional positive-loss pseudo-gradient, since Optax transforms
    apply a negative learning-rate scale to supplied gradients.
    """
    if losses.ndim != 1:
        raise ValueError(f"losses must be one-dimensional, got shape {losses.shape}")
    population_size = losses.shape[0]
    if population_size < 2:
        raise ValueError("at least two candidate losses are required")
    if not isinstance(sigma, float) or not math.isfinite(sigma) or sigma <= 0:
        raise ValueError(f"sigma must be a finite positive float, got {sigma!r}")
    if not jnp.issubdtype(losses.dtype, jnp.floating):
        raise TypeError(
            f"losses must be a floating-point array, got dtype {losses.dtype}"
        )
    centered = losses - jnp.mean(losses)
    return jnp.asarray(-(centered) / (population_size * sigma), dtype=losses.dtype)


def validate_losses(losses: Array, *, check_finite: bool = True) -> None:
    """Reject invalid per-member loss arrays before optimizer state advances.

    ``check_finite=False`` skips the device-to-host synchronization needed to
    turn the finite reduction into a Python exception.  Long-running compiled
    training loops can disable it after a validated warmup and keep the next
    generation queued on the GPU.
    """
    if losses.ndim != 1:
        raise ValueError(f"losses must be one-dimensional, got shape {losses.shape}")
    if losses.shape[0] < 2:
        raise ValueError("at least two candidate losses are required")
    if not jnp.issubdtype(losses.dtype, jnp.floating):
        raise TypeError(
            f"losses must be a floating-point array, got dtype {losses.dtype}"
        )
    if check_finite and not jnp.all(jnp.isfinite(losses)):
        raise ValueError("candidate losses must be finite")


def antithetical_pair_id(candidate_id: Array, population_size: int) -> Array:
    """Map a candidate id to its shared antithetical pair index in ``[0, N/2)``."""
    half = population_size // 2
    return jnp.asarray(candidate_id, dtype=jnp.int32) % half


def antithetical_sign(candidate_id: Array, population_size: int) -> Array:
    """``+1`` for the first half of the population, ``-1`` for the antithetical half."""
    half = population_size // 2
    return jnp.where(jnp.asarray(candidate_id, dtype=jnp.int32) < half, jnp.int32(1), jnp.int32(-1))


def shape_antithetical_loss(losses: Array) -> Array:
    """Per-pair weights ``sign(L_- - L_+)`` in ``{-1, 0, 1}`` (Eggroll Appendix H.2).

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
