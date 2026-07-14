"""Transactional ZeroGrad optimizer lifecycle over JAX parameter mappings and Optax."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import optax

from ._candidate import CandidateContext
from ._fitness import shape_centered_loss, validate_losses
from ._keys import candidate_key, step_key
from ._manifest import Manifest, ParameterTree
from ._replay import replay

Array = jax.Array

LossFn = Callable[[ParameterTree, CandidateContext, Any, Array], tuple[Array, Any]]


@dataclass(frozen=True, slots=True)
class ZeroGradState:
    """Immutable optimizer state: logical generation and opaque Optax state."""

    generation: int
    opt_state: Any


@dataclass(frozen=True, slots=True)
class StepMetrics:
    """Population diagnostics returned by ``ZeroGrad.step``."""

    generation: int
    mean_loss: float
    min_loss: float
    max_loss: float
    population_size: int


class ZeroGrad:
    """A drop-in zero-gradient optimizer using factor-only ES perturbations.

    Parameters:
        manifest: Explicit parameter identity and layout manifest.
        transform: An Optax ``GradientTransformation`` (e.g. ``optax.adamw(...)``).
        population_size: Number of candidates per generation.
        rank: Low-rank factor dimension for matrix/table layouts.
        sigma: Perturbation scale.
        seed: Deterministic base seed for replay identity.
        run_id: Stable run identifier for cross-process reproducibility.
    """

    def __init__(
        self,
        manifest: Manifest,
        transform: optax.GradientTransformation,
        population_size: int,
        rank: int,
        sigma: float,
        seed: int,
        run_id: str,
    ) -> None:
        if not isinstance(manifest, Manifest):
            raise TypeError("manifest must be a Manifest")
        if not isinstance(population_size, int) or isinstance(population_size, bool) or population_size < 2:
            raise ValueError("population_size must be an integer >= 2")
        if not isinstance(rank, int) or isinstance(rank, bool) or rank < 1:
            raise ValueError("rank must be a positive integer")
        if not isinstance(sigma, float) or sigma <= 0:
            raise ValueError("sigma must be a positive float")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError("seed must be an integer")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("run_id must be a non-empty string")
        self._manifest = manifest
        self._transform = transform
        self._population_size = population_size
        self._rank = rank
        self._sigma = sigma
        self._seed = seed
        self._run_id = run_id

    def init(self, params: ParameterTree) -> ZeroGradState:
        """Initialize optimizer state for the given parameter mapping."""
        self._manifest.validate(params)
        opt_state = self._transform.init(params)
        return ZeroGradState(generation=0, opt_state=opt_state)

    def step(
        self,
        state: ZeroGradState,
        params: ParameterTree,
        batch: Any,
        loss_fn: LossFn,
        *,
        rng: Array | None = None,
    ) -> tuple[ParameterTree, ZeroGradState, StepMetrics]:
        """Execute one transactional optimization generation.

        Generates deterministic candidates, evaluates their losses via ``loss_fn``,
        shapes them into centered-loss weights, replays factor-only pseudo-gradients,
        negates the descent direction into a conventional positive-loss gradient,
        applies the Optax transform, and returns new parameters, state, and metrics.
        """
        if not isinstance(state, ZeroGradState):
            raise TypeError("state must be a ZeroGradState")
        self._manifest.validate(params)
        if rng is None:
            rng = jax.random.key(0)

        generation = state.generation
        base_key = step_key(self._seed, self._run_id, generation, self._manifest.version)
        candidate_ids = jnp.arange(self._population_size, dtype=jnp.int32)

        def evaluate_candidate(candidate_id: Array) -> Array:
            ck = candidate_key(base_key, candidate_id)
            ctx = CandidateContext(self._manifest, ck, self._rank, self._sigma)
            result = loss_fn(params, ctx, batch, rng)
            loss, _aux = result
            return loss

        losses = jax.vmap(evaluate_candidate)(candidate_ids)
        validate_losses(losses)
        shaped = shape_centered_loss(losses, self._sigma)

        descent = replay(params, self._manifest, base_key, candidate_ids, shaped, self._rank)
        pseudo_grad = _build_pseudo_grad(descent, params)

        updates, new_opt_state = self._transform.update(pseudo_grad, state.opt_state, params)
        new_params = optax.apply_updates(params, updates)

        new_state = ZeroGradState(generation=generation + 1, opt_state=new_opt_state)
        metrics = StepMetrics(
            generation=generation,
            mean_loss=float(jnp.mean(losses)),
            min_loss=float(jnp.min(losses)),
            max_loss=float(jnp.max(losses)),
            population_size=self._population_size,
        )
        return new_params, new_state, metrics


def _build_pseudo_grad(descent: dict, params: ParameterTree) -> dict:
    """Negate the descent direction and fill zeros for non-manifest parameters."""
    result: dict = {}
    _apply_negation(params, (), descent, result)
    return result


def _apply_negation(params: ParameterTree, path: tuple[str, ...], descent: dict, out: dict) -> None:
    """Walk the parameter tree, negating manifest entries and zeroing others."""
    for key, value in params.items():
        current_path = path + (key,)
        if isinstance(value, dict):
            sub_out: dict = {}
            _apply_negation(value, current_path, descent, sub_out)
            out[key] = sub_out
        elif isinstance(value, jax.Array):
            node = descent
            found = True
            for part in current_path:
                if not isinstance(node, dict) or part not in node:
                    found = False
                    break
                node = node[part]
            if found and isinstance(node, jax.Array):
                out[key] = -node
            else:
                out[key] = jnp.zeros_like(value)
        else:
            out[key] = value
