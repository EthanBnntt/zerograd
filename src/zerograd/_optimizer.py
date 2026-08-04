"""Transactional ZeroGrad optimizer lifecycle over JAX parameter mappings and Optax."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from ._candidate import CandidateContext
from ._eggroll_h import (
    antithetical_pair_id,
    antithetical_sign,
    apply_bin_updates,
    shape_antithetical_loss,
    threshold_tree_for_manifest,
    update_alpha_schedule,
)
from ._fitness import shape_centered_loss, validate_losses
from ._integer import float_view_tree, snap_tree_to_integer
from ._keys import candidate_key, step_key
from ._manifest import Manifest, ParameterTree
from ._nnx import (
    apply_surgery,
    bind_candidate,
    disable_candidates,
    params_pure_dict,
    update_params,
)
from ._replay import replay, replay_integer

Array = jax.Array

# Legacy dict API: (params, CandidateContext, batch, rng) -> (loss, aux)
LossFn = Callable[[ParameterTree, CandidateContext, Any, Array], tuple[Array, Any]]
# NNX API: (model, batch) -> (loss, aux)  or  (model, batch, rng) -> (loss, aux)
ModelLossFn = Callable[..., tuple[Array, Any]]

_UNSET: Any = object()


def _coalesce(kw: Any, default: Any) -> Any:
    return default if kw is _UNSET else kw


@dataclass(frozen=True, slots=True)
class ZeroGradState:
    """Immutable optimizer state: logical generation and opaque Optax state."""

    generation: int
    opt_state: Any


@dataclass(frozen=True, slots=True)
class StepMetrics:
    """Population diagnostics returned by ``ZeroGrad.step``."""

    generation: int
    # Device scalar arrays are intentional: converting these to Python floats
    # inside every step serialized the entire GPU queue three times.  Callers
    # naturally synchronize when they print/log a metric.
    mean_loss: Array
    min_loss: Array
    max_loss: Array
    population_size: int


class ZeroGrad:
    """A drop-in zero-gradient optimizer using factor-only ES perturbations.

    **NNX (primary)**::

        opt = ZeroGrad(optax.adamw(1e-2), population_size=32, rank=4, sigma=0.1,
                       seed=0, run_id=\"exp\", candidate_chunk_size=1)  # low-VRAM
        state = opt.init(model)  # graph surgery + auto-manifest
        model, state, metrics = opt.step(state, model, batch, loss_fn)

    **Legacy dict + Manifest** (still supported)::

        opt = ZeroGrad(manifest, optax.adamw(1e-2), population_size=32, ...)
        state = opt.init(params)
        params, state, metrics = opt.step(state, params, batch, loss_fn)
    """

    def __init__(
        self,
        *args: Any,
        transform: optax.GradientTransformation | None = None,
        population_size: Any = _UNSET,
        rank: Any = _UNSET,
        sigma: Any = _UNSET,
        seed: Any = _UNSET,
        run_id: Any = _UNSET,
        manifest: Manifest | None = None,
        integer_es: bool = False,
        sigma_shift: int = 4,
        update_alpha: float = 1.0,
        alpha_decay: float = 0.015,
        int_bits: int = 8,
        candidate_chunk_size: int | None = None,
        int_bin_updates: bool = False,
        check_finite: bool = True,
    ) -> None:
        # Resolve NNX-style and legacy positional / keyword forms.
        if args and isinstance(args[0], Manifest):
            # Legacy: ZeroGrad(manifest, transform, population_size, rank, sigma, seed, run_id)
            # Also: ZeroGrad(manifest, transform, population_size=..., rank=..., ...)
            manifest = args[0]
            if len(args) == 7:
                _, transform, population_size, rank, sigma, seed, run_id = args
            elif len(args) >= 2:
                transform = args[1]
                pos = list(args[2:])

                def _pick(idx: int, kw: Any, default: Any) -> Any:
                    if idx < len(pos):
                        return pos[idx]
                    return _coalesce(kw, default)

                population_size = _pick(0, population_size, 32)
                rank = _pick(1, rank, 4)
                sigma = _pick(2, sigma, 0.01)
                seed = _pick(3, seed, 0)
                run_id = _pick(4, run_id, "zerograd")
            else:
                raise TypeError(
                    "legacy ZeroGrad(manifest, transform, ...) requires a transform "
                    "as the second argument"
                )
        elif args and isinstance(args[0], optax.GradientTransformation):
            # NNX: ZeroGrad(transform, population_size=..., rank=..., ...)
            if len(args) > 6:
                raise TypeError("too many positional arguments for ZeroGrad")
            transform = args[0]
            pos = list(args[1:])

            def _pick(idx: int, kw: Any, default: Any) -> Any:
                if idx < len(pos):
                    return pos[idx]
                return _coalesce(kw, default)

            population_size = _pick(0, population_size, 32)
            rank = _pick(1, rank, 4)
            sigma = _pick(2, sigma, 0.01)
            seed = _pick(3, seed, 0)
            run_id = _pick(4, run_id, "zerograd")
        elif args:
            raise TypeError(
                "first argument must be an Optax GradientTransformation or a Manifest, "
                f"got {type(args[0]).__name__}"
            )
        else:
            if transform is None and not integer_es:
                raise TypeError("transform is required unless integer_es=True")
            population_size = _coalesce(population_size, 32)
            rank = _coalesce(rank, 4)
            sigma = _coalesce(sigma, 0.01)
            seed = _coalesce(seed, 0)
            run_id = _coalesce(run_id, "zerograd")

        if integer_es and transform is None:
            transform = optax.identity()

        if manifest is not None and not isinstance(manifest, Manifest):
            raise TypeError("manifest must be a Manifest")
        if not isinstance(transform, optax.GradientTransformation):
            raise TypeError("transform must be an Optax GradientTransformation")
        if not isinstance(population_size, int) or isinstance(population_size, bool) or population_size < 2:
            raise ValueError("population_size must be an integer >= 2")
        if integer_es and population_size % 2 != 0:
            raise ValueError(
                f"integer_es (Appendix H antithetical pairs) requires even "
                f"population_size, got {population_size}"
            )
        if not isinstance(rank, int) or isinstance(rank, bool) or rank < 1:
            raise ValueError("rank must be a positive integer")
        if not isinstance(sigma_shift, int) or isinstance(sigma_shift, bool) or sigma_shift < 0:
            raise ValueError(f"sigma_shift must be a non-negative int, got {sigma_shift!r}")
        if integer_es:
            # σ = 2^{-σ̂} from Appendix H; keep a float sigma for any legacy call sites.
            sigma = float(2.0 ** (-sigma_shift))
        if not isinstance(sigma, float) or not math.isfinite(sigma) or sigma <= 0:
            raise ValueError("sigma must be a finite positive float")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError("seed must be an integer")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("run_id must be a non-empty string")
        if not isinstance(update_alpha, float) or not math.isfinite(update_alpha):
            raise ValueError("update_alpha must be a finite float")
        if not (0.0 < update_alpha <= 1.0):
            raise ValueError(f"update_alpha must be in (0, 1], got {update_alpha!r}")
        if not isinstance(alpha_decay, float) or not math.isfinite(alpha_decay) or alpha_decay < 0:
            raise ValueError(f"alpha_decay must be a non-negative float, got {alpha_decay!r}")
        if not isinstance(int_bits, int) or isinstance(int_bits, bool) or int_bits < 2:
            raise ValueError(f"int_bits must be an int >= 2, got {int_bits!r}")
        if candidate_chunk_size is not None and (
            not isinstance(candidate_chunk_size, int)
            or isinstance(candidate_chunk_size, bool)
            or candidate_chunk_size < 1
        ):
            raise ValueError(
                f"candidate_chunk_size must be None or an int >= 1, got {candidate_chunk_size!r}"
            )
        if not isinstance(check_finite, bool):
            raise TypeError(f"check_finite must be bool, got {check_finite!r}")

        self._manifest = manifest
        self._transform = transform
        self._population_size = population_size
        self._rank = rank
        self._sigma = sigma
        self._seed = seed
        self._run_id = run_id
        self._nnx_mode = False
        self._integer_es = bool(integer_es)
        self._sigma_shift = int(sigma_shift)
        self._update_alpha = float(update_alpha)
        self._alpha_decay = float(alpha_decay)
        self._int_bits = int(int_bits)
        # None → vmap whole population (fast, high VRAM). Int → jax.lax.map chunks
        # (slower, peak activations ≈ chunk_size × one forward).
        self._candidate_chunk_size = candidate_chunk_size
        self._check_finite = check_finite
        # Appendix H bin updates when integer_es uses the identity transform.
        # Mixed mode (ternary, or int_bin_updates=True): ±1 bins on integer leaves,
        # Adam on float leaves — needed so int8 LUTs/kernels move under small Adam LRs.
        self._bin_updates = bool(integer_es) and _is_identity_transform(transform)
        self._ternary_bins = bool(integer_es) and self._int_bits == 2
        self._int_bin_updates = bool(integer_es) and (
            self._ternary_bins or (bool(int_bin_updates) and not self._bin_updates)
        )

    @classmethod
    def from_manifest(
        cls,
        manifest: Manifest,
        transform: optax.GradientTransformation,
        population_size: int,
        rank: int,
        sigma: float,
        seed: int,
        run_id: str,
    ) -> ZeroGrad:
        """Construct a dict-param ZeroGrad with an explicit Manifest (legacy API)."""
        return cls(manifest, transform, population_size, rank, sigma, seed, run_id)

    def init(self, model_or_params: nnx.Module | ParameterTree) -> ZeroGradState:
        """Initialize optimizer state.

        For an ``nnx.Module``, performs graph surgery (factor-aware layers) and
        builds an auto-manifest when none was provided. For a parameter dict,
        validates the explicit manifest.
        """
        if isinstance(model_or_params, nnx.Module):
            from ._nnx import ZeroGradSlot

            model = model_or_params
            already = hasattr(model, "zg_slot") and isinstance(model.zg_slot, ZeroGradSlot)
            if not already:
                model, auto_manifest = apply_surgery(
                    model,
                    rank=self._rank,
                    sigma=self._sigma,
                    sigma_shift=self._sigma_shift,
                    integer_es=self._integer_es,
                )
                # Surgery owns group identity; ignore any pre-set manifest so layer
                # group strings stay consistent with Manifest.entries.
                self._manifest = auto_manifest
            else:
                if self._manifest is None:
                    if model.zg_slot.manifest is None:
                        raise ValueError(
                            "model is already surged but has no manifest; "
                            "re-init from an unsurgered module"
                        )
                    self._manifest = model.zg_slot.manifest
                else:
                    model.zg_slot.manifest = self._manifest
            model.zg_slot.rank = self._rank
            model.zg_slot.sigma = self._sigma
            model.zg_slot.sigma_shift = self._sigma_shift
            model.zg_slot.integer_es = self._integer_es
            self._nnx_mode = True
            assert self._manifest is not None
            params = params_pure_dict(model)
            self._manifest.validate(params)
            if self._bin_updates:
                opt_state = None
            else:
                # Optax moments need float; integer leaves are viewed as float32.
                opt_state = self._transform.init(float_view_tree(params))
            return ZeroGradState(generation=0, opt_state=opt_state)

        if self._manifest is None:
            raise ValueError(
                "dict-parameter mode requires an explicit Manifest "
                "(use ZeroGrad.from_manifest(...) or manifest=...)"
            )
        self._nnx_mode = False
        self._manifest.validate(model_or_params)
        params = _materialize_tree(model_or_params)
        if self._bin_updates:
            opt_state = None
        else:
            opt_state = self._transform.init(float_view_tree(params))
        return ZeroGradState(generation=0, opt_state=opt_state)

    # ── Read-only configuration access (for distributed workers) ──────────────

    @property
    def population_size(self) -> int:
        """Number of candidates per generation."""
        return self._population_size

    @property
    def rank(self) -> int:
        """Low-rank factor dimension."""
        return self._rank

    @property
    def sigma(self) -> float:
        """Perturbation scale."""
        return self._sigma

    @property
    def manifest(self) -> Manifest:
        """Parameter manifest (set after ``init`` for NNX auto-manifest)."""
        if self._manifest is None:
            raise RuntimeError("manifest is not available before init in NNX mode")
        return self._manifest

    @property
    def seed(self) -> int:
        """Base seed for replay identity."""
        return self._seed

    @property
    def run_id(self) -> str:
        """Stable run identifier."""
        return self._run_id

    def evaluate_shard(
        self,
        model_or_params: nnx.Module | ParameterTree,
        generation: int,
        loss_fn: LossFn | ModelLossFn,
        batch: Any,
        candidate_ids: Array,
        *,
        rng: Array | None = None,
    ) -> Array:
        """Evaluate a subset of candidates and return their losses."""
        if isinstance(model_or_params, nnx.Module):
            return self._evaluate_shard_nnx(
                model_or_params, generation, loss_fn, batch, candidate_ids, rng=rng
            )
        assert self._manifest is not None
        self._manifest.validate(model_or_params)
        params = _materialize_tree(model_or_params)
        return self._evaluate_shard_dict(params, generation, loss_fn, batch, candidate_ids, rng=rng)

    def _evaluate_shard_dict(
        self,
        params: ParameterTree,
        generation: int,
        loss_fn: LossFn,
        batch: Any,
        candidate_ids: Array,
        *,
        rng: Array | None = None,
    ) -> Array:
        """Evaluate candidates (``vmap`` or chunked ``lax.map``) on the full batch."""
        if rng is None:
            rng = jax.random.key(0)
        assert self._manifest is not None
        base_key = step_key(self._seed, self._run_id, generation, self._manifest.version)

        def evaluate_candidate(candidate_id: Array) -> Array:
            ck = candidate_key(base_key, candidate_id)
            candidate_rng = jax.random.fold_in(rng, candidate_id)
            ctx = CandidateContext(self._manifest, ck, self._rank, self._sigma)
            loss, _aux = loss_fn(params, ctx, batch, candidate_rng)
            return loss

        return _map_candidates(
            evaluate_candidate, candidate_ids, chunk_size=self._candidate_chunk_size
        )

    @property
    def integer_es(self) -> bool:
        """Whether Appendix H pure-integer ES (antithetical + bin updates) is enabled."""
        return self._integer_es

    @property
    def sigma_shift(self) -> int:
        """Appendix H ``σ̂`` with ``σ = 2^{-σ̂}``."""
        return self._sigma_shift

    @property
    def candidate_chunk_size(self) -> int | None:
        """Candidate eval chunk size; ``None`` means full-population ``vmap``."""
        return self._candidate_chunk_size

    def _evaluate_shard_nnx(
        self,
        model: nnx.Module,
        generation: int,
        loss_fn: ModelLossFn,
        batch: Any,
        candidate_ids: Array,
        *,
        rng: Array | None = None,
    ) -> Array:
        """Evaluate candidates (``vmap`` or chunked ``lax.map``) on the full batch.

        Vmapped over candidates so population eval is one dense GPU program,
        not a Python loop of small forwards.
        """
        if rng is None:
            rng = jax.random.key(0)
        assert self._manifest is not None
        base_key = step_key(self._seed, self._run_id, generation, self._manifest.version)
        graphdef, state = nnx.split(model)
        pop = self._population_size

        def evaluate_candidate(candidate_id: Array) -> Array:
            if self._integer_es:
                pair = antithetical_pair_id(candidate_id, pop)
                sign = antithetical_sign(candidate_id, pop)
                ck = candidate_key(base_key, pair)
            else:
                ck = candidate_key(base_key, candidate_id)
                sign = 1
            candidate_rng = jax.random.fold_in(rng, candidate_id)
            bound = bind_candidate(graphdef, state, ck, enabled=True, factor_sign=sign)
            loss, _aux = _call_model_loss(loss_fn, bound, batch, candidate_rng)
            return loss

        return _map_candidates(
            evaluate_candidate, candidate_ids, chunk_size=self._candidate_chunk_size
        )

    def _evaluate_antithetical_nnx(
        self,
        model: nnx.Module,
        generation: int,
        loss_fn: ModelLossFn,
        batch: Any,
        *,
        rng: Array,
    ) -> Array:
        """Evaluate the full integer-ES population as dense ``[pair, sign]`` batches.

        The perturbation key depends on ``pair`` while only ``factor_sign`` and
        the model activations vary across the inner sign axis.  Nesting the sign
        vmap lets JAX hoist factor generation out of that axis, so antithetical
        candidates share their A/B factors instead of regenerating identical
        random tensors twice.
        """
        assert self._manifest is not None
        pop = self._population_size
        half = pop // 2
        base_key = step_key(self._seed, self._run_id, generation, self._manifest.version)
        graphdef, model_state = nnx.split(model)
        signs = jnp.asarray((1, -1), dtype=jnp.int32)
        sign_offsets = jnp.asarray((0, half), dtype=jnp.int32)

        def evaluate_pair(pair_id: Array) -> Array:
            ck = candidate_key(base_key, pair_id)

            def evaluate_sign(sign_and_offset: tuple[Array, Array]) -> Array:
                sign, offset = sign_and_offset
                candidate_id = pair_id + offset
                candidate_rng = jax.random.fold_in(rng, candidate_id)
                bound = bind_candidate(
                    graphdef, model_state, ck, enabled=True, factor_sign=sign
                )
                loss, _aux = _call_model_loss(loss_fn, bound, batch, candidate_rng)
                return loss

            return jax.vmap(evaluate_sign)((signs, sign_offsets))

        # candidate_chunk_size is expressed in candidates; one pair contributes
        # two candidates to the inner dense sign axis.
        pair_chunk = (
            None
            if self._candidate_chunk_size is None
            else max(1, self._candidate_chunk_size // 2)
        )
        pair_losses = _map_candidates(
            evaluate_pair,
            jnp.arange(half, dtype=jnp.int32),
            chunk_size=pair_chunk,
        )
        # Preserve the public ordering: all + candidates, then all - candidates.
        return jnp.concatenate((pair_losses[:, 0], pair_losses[:, 1]), axis=0)

    def _jit_update_bin_params(self, params: ParameterTree, thresholds: ParameterTree):
        """JIT the bin-update path once per params shape (cached on the instance).

        Thresholds are folded as static Python ints (not traced) so the jitted
        function only takes array arguments.
        """
        if not hasattr(self, "_bin_update_jit_cache"):
            self._bin_update_jit_cache: dict = {}

        sig = _tree_sig(params)
        jit_fn = self._bin_update_jit_cache.get(sig)
        if jit_fn is None:

            def _update(params_p, evidence_p):
                return apply_bin_updates(params_p, evidence_p, thresholds)

            jit_fn = jax.jit(_update)
            self._bin_update_jit_cache[sig] = jit_fn
        return jit_fn

    def _jit_integer_bin_step(self, params: ParameterTree):
        """Compile replay + thresholded bin update as one device program.

        Keeping replay and update in separate eager dispatches materialized an
        int32 evidence tree (roughly 4× the int8 model size) and launched one
        program per manifest leaf.  The combined executable can schedule leaves
        together and consume each evidence tensor directly into its bin update.
        """
        if not hasattr(self, "_integer_bin_step_jit_cache"):
            self._integer_bin_step_jit_cache: dict = {}

        sig = _tree_sig(params)
        jit_fn = self._integer_bin_step_jit_cache.get(sig)
        if jit_fn is None:
            assert self._manifest is not None
            manifest = self._manifest
            rank = self._rank
            half = self._population_size // 2
            pair_ids = jnp.arange(half, dtype=jnp.int32)
            from ._integer import qrange

            qmin, qmax = qrange(self._int_bits)

            def replay_and_update(params_p, losses_p, base_key_p, thresholds_p):
                shaped_p = shape_antithetical_loss(losses_p)
                evidence_p = replay_integer(
                    params_p,
                    manifest,
                    base_key_p,
                    pair_ids,
                    shaped_p,
                    rank,
                )
                return apply_bin_updates(
                    params_p,
                    evidence_p,
                    thresholds_p,
                    qmin=qmin,
                    qmax=qmax,
                )

            jit_fn = jax.jit(replay_and_update)
            self._integer_bin_step_jit_cache[sig] = jit_fn
        return jit_fn

    def step_from_losses(
        self,
        state: ZeroGradState,
        model_or_params: nnx.Module | ParameterTree,
        losses: Array,
    ) -> tuple[Any, ZeroGradState, StepMetrics]:
        """Complete one generation using pre-computed candidate losses."""
        if not isinstance(state, ZeroGradState):
            raise TypeError("state must be a ZeroGradState")
        self._check_losses(losses)

        if isinstance(model_or_params, nnx.Module):
            model = model_or_params
            assert self._manifest is not None
            params = params_pure_dict(model)
            self._manifest.validate(params)
            new_params, new_state, metrics = self._step_from_losses(state, params, losses)
            update_params(model, new_params)
            disable_candidates(model)
            return model, new_state, metrics

        assert self._manifest is not None
        self._manifest.validate(model_or_params)
        params = _materialize_tree(model_or_params)
        return self._step_from_losses(state, params, losses)

    def _check_losses(self, losses: Array) -> None:
        """Validate candidate losses (dtype, finiteness, count)."""
        validate_losses(losses, check_finite=self._check_finite)
        if losses.shape[0] != self._population_size:
            raise ValueError(
                f"losses must have {self._population_size} entries, got {losses.shape[0]}"
            )

    def _step_from_losses(
        self,
        state: ZeroGradState,
        params: ParameterTree,
        losses: Array,
    ) -> tuple[ParameterTree, ZeroGradState, StepMetrics]:
        """Internal fast-path generation completion that skips re-validation."""
        assert self._manifest is not None
        generation = state.generation
        base_key = step_key(self._seed, self._run_id, generation, self._manifest.version)

        if self._bin_updates or self._int_bin_updates:
            half = self._population_size // 2
            pair_ids = jnp.arange(half, dtype=jnp.int32)
            alpha = update_alpha_schedule(
                generation, base=self._update_alpha, decay=self._alpha_decay
            )
            alpha_clip = min(max(alpha, 1e-6), 1.0 - 1e-6)
            # Layout-scaled thresholds: vectors use 16√N, matrices/tables 16·16·√N.
            thresholds = threshold_tree_for_manifest(
                params,
                self._manifest,
                alpha=alpha_clip,
                num_directions=half,
            )
            if self._bin_updates:
                # Pure integer path: compile replay and update together so the
                # full int32 evidence tree never crosses a dispatch boundary.
                update_fn = self._jit_integer_bin_step(params)
                new_params = update_fn(params, losses, base_key, thresholds)
                new_opt_state = None
            else:
                shaped = shape_antithetical_loss(losses)
                evidence = replay_integer(
                    params, self._manifest, base_key, pair_ids, shaped, self._rank
                )
                # Mixed path also needs float-leaf evidence below.
                update_fn = self._jit_update_bin_params(params, thresholds)
                new_params = update_fn(params, evidence)
                # Mixed int bins + AdamW: bin-flip integer leaves, Adam float leaves.
                shaped_f = shape_centered_loss(losses, self._sigma)
                pair_weights = shaped_f[:half] - shaped_f[half:]
                descent = replay_integer(
                    params, self._manifest, base_key, pair_ids, pair_weights, self._rank
                )
                pseudo_grad = _build_pseudo_grad(descent, params)
                pseudo_grad = _mask_integer_leaves(pseudo_grad, new_params)
                float_params = float_view_tree(new_params)
                updates, new_opt_state = self._transform.update(
                    pseudo_grad, state.opt_state, float_params
                )
                new_float = optax.apply_updates(float_params, updates)
                new_params = _merge_keep_integers(new_params, new_float)
        elif self._integer_es:
            half = self._population_size // 2
            pair_ids = jnp.arange(half, dtype=jnp.int32)
            shaped = shape_centered_loss(losses, self._sigma)
            pair_weights = shaped[:half] - shaped[half:]
            descent = replay_integer(
                params, self._manifest, base_key, pair_ids, pair_weights, self._rank
            )
            pseudo_grad = _build_pseudo_grad(descent, params)
            float_params = float_view_tree(params)
            updates, new_opt_state = self._transform.update(
                pseudo_grad, state.opt_state, float_params
            )
            new_float = optax.apply_updates(float_params, updates)
            new_params = snap_tree_to_integer(new_float, params, bits=self._int_bits)
        else:
            candidate_ids = jnp.arange(self._population_size, dtype=jnp.int32)
            shaped = shape_centered_loss(losses, self._sigma)
            descent = replay(params, self._manifest, base_key, candidate_ids, shaped, self._rank)
            pseudo_grad = _build_pseudo_grad(descent, params)

            float_params = float_view_tree(params)
            updates, new_opt_state = self._transform.update(
                pseudo_grad, state.opt_state, float_params
            )
            new_float = optax.apply_updates(float_params, updates)
            new_params = snap_tree_to_integer(new_float, params)

        new_state = ZeroGradState(generation=generation + 1, opt_state=new_opt_state)
        metrics = StepMetrics(
            generation=generation,
            mean_loss=jnp.mean(losses),
            min_loss=jnp.min(losses),
            max_loss=jnp.max(losses),
            population_size=self._population_size,
        )
        return new_params, new_state, metrics

    def step(
        self,
        state: ZeroGradState,
        model_or_params: nnx.Module | ParameterTree,
        batch: Any,
        loss_fn: LossFn | ModelLossFn,
        *,
        rng: Array | None = None,
    ) -> tuple[Any, ZeroGradState, StepMetrics]:
        """Execute one transactional optimization generation."""
        if not isinstance(state, ZeroGradState):
            raise TypeError("state must be a ZeroGradState")
        if rng is None:
            rng = jax.random.key(0)

        generation = state.generation
        candidate_ids = jnp.arange(self._population_size, dtype=jnp.int32)

        if isinstance(model_or_params, nnx.Module):
            model = model_or_params
            if self._integer_es:
                losses = self._evaluate_antithetical_nnx(
                    model, generation, loss_fn, batch, rng=rng
                )
            else:
                losses = self._evaluate_shard_nnx(
                    model, generation, loss_fn, batch, candidate_ids, rng=rng
                )
            return self.step_from_losses(state, model, losses)

        assert self._manifest is not None
        self._manifest.validate(model_or_params)
        params = _materialize_tree(model_or_params)
        losses = self._evaluate_shard_dict(
            params, generation, loss_fn, batch, candidate_ids, rng=rng
        )
        self._check_losses(losses)
        return self._step_from_losses(state, params, losses)


def _map_candidates(
    evaluate_candidate: Callable[[Array], Array],
    candidate_ids: Array,
    *,
    chunk_size: int | None,
) -> Array:
    """Map ``evaluate_candidate`` over ids.

    ``chunk_size=None`` uses ``vmap`` (all candidates live at once). An int uses
    ``jax.lax.map(..., batch_size=chunk_size)`` so peak activations scale with
    the chunk, not the full population — needed for CNN-sized forwards on ≤16GB.
    """
    if chunk_size is None:
        return jax.vmap(evaluate_candidate)(candidate_ids)
    return jax.lax.map(evaluate_candidate, candidate_ids, batch_size=int(chunk_size))


def _is_identity_transform(transform: optax.GradientTransformation) -> bool:
    """True when ``transform`` is Optax identity (passes gradients through unchanged)."""
    probe = jnp.ones((2,), dtype=jnp.float32)
    state = transform.init(probe)
    updates, _ = transform.update(probe, state, probe)
    # identity: updates == grads; adam/sgd: updates differ in scale/sign handling
    return bool(jnp.allclose(updates, probe))


def _call_model_loss(
    loss_fn: ModelLossFn, model: nnx.Module, batch: Any, rng: Array
) -> tuple[Array, Any]:
    """Call ``loss_fn(model, batch)`` or ``loss_fn(model, batch, rng)``."""
    try:
        return loss_fn(model, batch, rng)
    except TypeError:
        return loss_fn(model, batch)


def _tree_sig(tree: Any) -> Any:
    """Lightweight shape/dtype signature for caching per-tree jits (hashable)."""
    if isinstance(tree, Mapping):
        return tuple(
            (str(k), _tree_sig(v))
            for k, v in sorted(tree.items(), key=lambda kv: str(kv[0]))
        )
    if isinstance(tree, jax.Array):
        return (tuple(tree.shape), str(tree.dtype))
    return type(tree).__name__


def _materialize_tree(params: ParameterTree) -> dict:
    """Return a plain-``dict`` copy of a parameter tree."""
    out: dict = {}
    for key, value in params.items():
        if isinstance(value, Mapping):
            out[key] = _materialize_tree(value)
        else:
            out[key] = value
    return out


def _build_pseudo_grad(descent: dict, params: ParameterTree) -> dict:
    """Negate the descent direction and fill zeros for non-manifest parameters."""
    result: dict = {}
    _apply_negation(params, (), descent, result)
    return result


def _mask_integer_leaves(grads: dict, params: ParameterTree) -> dict:
    """Zero Optax grads on integer leaves (already handled by bin updates)."""
    out: dict = {}
    for key, value in params.items():
        g = grads.get(key) if isinstance(grads, dict) else None
        if isinstance(value, Mapping):
            out[key] = _mask_integer_leaves(g if isinstance(g, dict) else {}, value)
        elif isinstance(value, jax.Array) and jnp.issubdtype(value.dtype, jnp.integer):
            out[key] = jnp.zeros_like(value, dtype=jnp.float32)
        elif isinstance(g, jax.Array):
            out[key] = g
        elif isinstance(value, jax.Array):
            out[key] = jnp.zeros_like(value, dtype=jnp.float32)
        else:
            out[key] = value
    return out


def _merge_keep_integers(int_params: ParameterTree, float_updated: ParameterTree) -> dict:
    """Keep bin-updated integer leaves; take Adam-updated float leaves."""
    out: dict = {}
    for key, value in int_params.items():
        other = float_updated[key] if isinstance(float_updated, dict) else None
        if isinstance(value, Mapping) and isinstance(other, Mapping):
            out[key] = _merge_keep_integers(value, other)
        elif isinstance(value, jax.Array) and jnp.issubdtype(value.dtype, jnp.integer):
            out[key] = value
        elif isinstance(other, jax.Array):
            out[key] = other.astype(value.dtype) if isinstance(value, jax.Array) else other
        else:
            out[key] = value
    return out


def _apply_negation(params: ParameterTree, path: tuple[str, ...], descent: dict, out: dict) -> None:
    """Walk the parameter tree, negating manifest entries and zeroing others."""
    for key, value in params.items():
        current_path = path + (key,)
        if isinstance(value, Mapping):
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