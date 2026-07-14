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
    centered = losses - jnp.mean(losses)
    return jnp.asarray(-(centered) / (population_size * sigma), dtype=losses.dtype)


def validate_losses(losses: Array) -> None:
    """Reject non-scalar per-member losses before optimizer state advances."""
    if losses.ndim != 1:
        raise ValueError(f"losses must be one-dimensional, got shape {losses.shape}")
    if losses.shape[0] < 2:
        raise ValueError("at least two candidate losses are required")
    if not jnp.all(jnp.isfinite(losses)):
        raise ValueError("candidate losses must be finite")
