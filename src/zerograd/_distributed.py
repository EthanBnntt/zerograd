"""Multi-device distributed evaluation for ZeroGrad.

The ES population is embarrassingly parallel: each candidate's loss is
computed independently, and only the 1D fitness array needs to be shared
between workers.  This module provides:

- ``DeviceShard``: evaluates a subset of candidates on one JAX device.
- ``DistributedZeroGrad``: coordinates multiple shards, gathers fitnesses,
  and drives the optimizer step.

Workers can be mixed across CPU and GPU, or multiple shards can share a
single GPU.  Population is partitioned across devices using **weights** —
a fast GPU can receive more candidates than a slow CPU.  Weights can be
set manually or auto-calibrated by timing each device.

Only the 1D loss arrays cross device boundaries — typically a few hundred
floats per step, negligible overhead.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Sequence, TypeAlias, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from ._nnx import params_pure_dict, update_params
from ._optimizer import ModelLossFn, StepMetrics, ZeroGrad, ZeroGradState

Array = jax.Array
Device: TypeAlias = Any  # jax.Device is nanobind; not usable in type expressions

_T = TypeVar("_T")
_R = TypeVar("_R")


def gather_shard_losses(
    executor: ThreadPoolExecutor,
    shards: Sequence[_T],
    evaluate_fn: Callable[[_T], _R],
    *,
    concat: Callable[[list[_R]], _R],
    is_empty: Callable[[_T], bool] | None = None,
) -> _R:
    """Submit ``evaluate_fn(shard)`` concurrently for each shard, then concatenate in order.

    Shards for which ``is_empty`` returns ``True`` are skipped entirely (no
    future submitted, nothing to gather) — used by :class:`DistributedZeroGrad`
    to avoid evaluating a zero-candidate partition.
    """
    futures = [
        None if is_empty is not None and is_empty(shard) else executor.submit(evaluate_fn, shard)
        for shard in shards
    ]
    parts = [future.result() for future in futures if future is not None]
    return concat(parts)


@dataclass(frozen=True, slots=True)
class ShardResult:
    """One worker's contribution: candidate IDs and their losses."""

    candidate_ids: Array
    losses: Array


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    """Per-device timing from auto-calibration."""

    device: Device
    name: str
    num_candidates: int
    elapsed_seconds: float
    per_candidate_seconds: float


def compute_partition_sizes(population: int, weights: list[float]) -> list[int]:
    """Split ``population`` into len(weights) parts proportional to weights.

    Uses the largest remainder method so the sum always equals ``population``
    exactly, even when the division isn't clean.  Devices with zero weight
    receive zero candidates.
    """
    if not isinstance(population, int) or isinstance(population, bool) or population < 1:
        raise ValueError(f"population must be a positive integer, got {population!r}")
    if not weights:
        raise ValueError("weights must be non-empty")
    if any(w < 0 for w in weights):
        raise ValueError("weights must be non-negative")
    total = sum(weights)
    if total == 0:
        raise ValueError("at least one weight must be positive")

    quotas = [population * w / total for w in weights]
    sizes = [int(q) for q in quotas]
    remainder = population - sum(sizes)

    # Distribute remainder to largest fractional parts
    frac_order = sorted(
        range(len(weights)),
        key=lambda i: quotas[i] - int(quotas[i]),
        reverse=True,
    )
    for i in range(remainder):
        sizes[frac_order[i]] += 1

    return sizes


def split_candidate_ids(population: int, sizes: Sequence[int]) -> list[Array]:
    """Split ``0..population-1`` into contiguous shards matching ``sizes``."""
    all_ids = jnp.arange(population, dtype=jnp.int32)
    if len(sizes) <= 1:
        return [all_ids]
    split_points = jnp.cumsum(jnp.asarray(sizes[:-1]))
    return list(jnp.split(all_ids, split_points))


class DeviceShard:
    """Evaluate a population shard on a specific JAX device.

    Parameters:
        device: The JAX device to evaluate on (CPU or GPU).
        optimizer: The ZeroGrad optimizer (provides evaluate_shard).
        loss_fn: The loss function to evaluate candidates with.
        name: Human-readable label for logging.
    """

    def __init__(
        self,
        device: Device,
        optimizer: ZeroGrad,
        loss_fn: ModelLossFn,
        name: str = "",
    ) -> None:
        self.device = device
        self._optimizer = optimizer
        self._loss_fn = loss_fn
        self.name = name or f"shard-{device.platform}"

    def evaluate(
        self,
        params: Any,
        batch: Any,
        candidate_ids: Array,
        generation: int,
        rng: Array | None = None,
    ) -> ShardResult:
        """Evaluate the assigned candidates on this shard's device."""
        if rng is None:
            rng = jax.random.key(0)

        # NNX modules are not device_put-friendly as a whole; put batch/ids only.
        if isinstance(params, nnx.Module):
            batch_d = jax.device_put(batch, self.device)
            ids_d = jax.device_put(candidate_ids, self.device)
            rng_d = jax.device_put(rng, self.device)
            with jax.default_device(self.device):
                losses = self._optimizer.evaluate_shard(
                    params, generation, self._loss_fn, batch_d, ids_d, rng=rng_d,
                )
        else:
            params_d = jax.device_put(params, self.device)
            batch_d = jax.device_put(batch, self.device)
            ids_d = jax.device_put(candidate_ids, self.device)
            rng_d = jax.device_put(rng, self.device)
            losses = self._optimizer.evaluate_shard(
                params_d, generation, self._loss_fn, batch_d, ids_d, rng=rng_d,
            )
        return ShardResult(candidate_ids=candidate_ids, losses=losses)

    @property
    def num_candidates(self) -> int:
        """Number of candidates assigned to this shard."""
        ids = self._candidate_ids
        return int(ids.shape[0]) if ids is not None else 0

    # Set by the coordinator when partitioning changes
    _candidate_ids: Array | None = None


class DistributedZeroGrad:
    """Multi-device ZeroGrad coordinator with weighted partitioning.

    Splits the population across devices proportional to ``weights``.
    A fast GPU with weight 4 receives four times as many candidates as a
    slow CPU with weight 1.  Weights can be set manually or auto-calibrated
    by timing each device's per-candidate evaluation speed.

    Each step:
    1. Each shard evaluates its assigned candidates on its device (concurrently).
    2. Losses are gathered and concatenated in candidate-ID order.
    3. ``step_from_losses`` completes the generation on the coordinator device.

    Parameters:
        optimizer: The ZeroGrad optimizer to coordinate.
        devices: List of JAX devices to evaluate on.
        loss_fn: The loss function for candidate evaluation.
        weights: Relative compute weights, one per device.  ``None`` means
            equal weights (even split).  Example: ``[1, 4]`` gives the second
            device 4× as many candidates as the first.
        coordinator_device: Device for the step_from_losses computation
            (defaults to the first device).
    """

    def __init__(
        self,
        optimizer: ZeroGrad,
        devices: list[Device],
        loss_fn: ModelLossFn,
        weights: list[float] | None = None,
        coordinator_device: Device | None = None,
    ) -> None:
        if not devices:
            raise ValueError("at least one device is required")
        if weights is not None and len(weights) != len(devices):
            raise ValueError(
                f"weights length ({len(weights)}) must match devices ({len(devices)})"
            )

        self._optimizer = optimizer
        self._devices = devices
        self._coordinator_device = coordinator_device or devices[0]
        self._weights = weights or [1.0] * len(devices)

        # Create one shard per device
        self._shards: list[DeviceShard] = [
            DeviceShard(dev, optimizer, loss_fn, name=f"worker-{i}-{dev.platform}")
            for i, dev in enumerate(devices)
        ]

        # Compute initial partition
        self._partition_sizes = compute_partition_sizes(
            optimizer.population_size, self._weights,
        )
        self._apply_partition()

        # Thread pool for concurrent evaluation
        self._executor = ThreadPoolExecutor(max_workers=len(devices))

    def shutdown(self) -> None:
        """Shut down the evaluation thread pool and release worker threads.

        Idempotent; safe to call multiple times.  After shutdown the
        coordinator can no longer ``step``.
        """
        executor = self._executor
        if executor is not None:
            executor.shutdown(wait=False)
            self._executor = None

    def __enter__(self) -> DistributedZeroGrad:
        return self

    def __exit__(self, *exc: object) -> bool:
        self.shutdown()
        return False

    def __del__(self) -> None:
        try:
            self.shutdown()
        except Exception:
            pass

    def _apply_partition(self) -> None:
        """Split candidate IDs according to current partition sizes and assign to shards."""
        for shard, ids in zip(
            self._shards,
            split_candidate_ids(self._optimizer.population_size, self._partition_sizes),
        ):
            shard._candidate_ids = ids

    def init(self, model_or_params: Any) -> ZeroGradState:
        """Initialize optimizer state (dict params or nnx.Module)."""
        return self._optimizer.init(model_or_params)

    def step(
        self,
        state: ZeroGradState,
        model_or_params: Any,
        batch: Any,
        *,
        rng: Array | None = None,
    ) -> tuple[Any, ZeroGradState, StepMetrics]:
        """Execute one distributed optimization generation.

        Evaluates each shard concurrently, gathers losses in candidate-ID
        order, and completes the step on the coordinator device.
        """
        if rng is None:
            rng = jax.random.key(0)
        if self._executor is None:
            raise RuntimeError("coordinator has been shut down")

        gen = state.generation

        def _evaluate(shard: DeviceShard) -> Array:
            candidate_ids = shard._candidate_ids
            assert candidate_ids is not None
            result = shard.evaluate(model_or_params, batch, candidate_ids, gen, rng)
            return jax.device_put(result.losses, self._coordinator_device)

        def _shard_is_empty(shard: DeviceShard) -> bool:
            ids = shard._candidate_ids
            return ids is None or ids.shape[0] == 0

        # Shards already hold sequential candidate ID ranges, so gathering in
        # shard order reconstructs the population's loss array directly.
        losses = gather_shard_losses(
            self._executor,
            self._shards,
            _evaluate,
            concat=jnp.concatenate,
            is_empty=_shard_is_empty,
        )

        if isinstance(model_or_params, nnx.Module):
            return self._optimizer.step_from_losses(state, model_or_params, losses)

        params_c = jax.device_put(model_or_params, self._coordinator_device)
        return self._optimizer.step_from_losses(state, params_c, losses)

    def calibrate(
        self,
        model_or_params: Any,
        batch: Any,
        *,
        warmup: int = 1,
        trials: int = 3,
        rng: Array | None = None,
    ) -> list[CalibrationResult]:
        """Auto-calibrate weights by timing each device."""
        if rng is None:
            rng = jax.random.key(0)

        pop = self._optimizer.population_size
        calib_ids = jnp.arange(min(pop, 4), dtype=jnp.int32)
        gen = 0

        results: list[CalibrationResult] = []

        for shard in self._shards:
            for _ in range(warmup):
                r = shard.evaluate(model_or_params, batch, calib_ids, gen, rng)
                jax.block_until_ready(r.losses)

            t0 = time.perf_counter()
            for _ in range(trials):
                r = shard.evaluate(model_or_params, batch, calib_ids, gen, rng)
                jax.block_until_ready(r.losses)
            elapsed = (time.perf_counter() - t0) / trials
            per_candidate = elapsed / len(calib_ids)

            results.append(CalibrationResult(
                device=shard.device,
                name=shard.name,
                num_candidates=len(calib_ids),
                elapsed_seconds=elapsed,
                per_candidate_seconds=per_candidate,
            ))

        new_weights = [1.0 / r.per_candidate_seconds for r in results]
        self._weights = new_weights
        self._partition_sizes = compute_partition_sizes(pop, new_weights)
        self._apply_partition()

        return results

    @property
    def shards(self) -> list[DeviceShard]:
        """The device shards managed by this coordinator."""
        return self._shards

    @property
    def weights(self) -> list[float]:
        """Current compute weights, one per device."""
        return list(self._weights)

    @property
    def partition_sizes(self) -> list[int]:
        """Number of candidates assigned to each shard."""
        return list(self._partition_sizes)


@dataclass(slots=True)
class _DeviceReplica:
    device: Device
    optimizer: ZeroGrad
    model: Any
    state: ZeroGradState
    candidate_ids: np.ndarray


class ReplicatedDistributedZeroGrad:
    """Seed-replicated, gradient-free ES across local accelerator devices.

    Each GPU owns an independently initialized model/optimizer replica. Candidate
    shards are evaluated concurrently; only the scalar loss vector is gathered
    and broadcast. Every replica deterministically replays the same update, so
    parameters remain synchronized without parameter, activation, or gradient
    communication.
    """

    def __init__(
        self,
        *,
        devices: list[Device],
        optimizer_factory: Callable[[], ZeroGrad],
        model_factory: Callable[[], Any],
        loss_fn: ModelLossFn,
        weights: list[float] | None = None,
    ) -> None:
        if not devices:
            raise ValueError("at least one device is required")
        if weights is not None and len(weights) != len(devices):
            raise ValueError(
                f"weights length ({len(weights)}) must match devices ({len(devices)})"
            )

        self._devices = list(devices)
        self._loss_fn = loss_fn
        self._executor: ThreadPoolExecutor | None = ThreadPoolExecutor(
            max_workers=len(devices)
        )
        self._replicas: list[_DeviceReplica] = []

        population: int | None = None
        for device in self._devices:
            with jax.default_device(device):
                optimizer = optimizer_factory()
                model = model_factory()
                state = optimizer.init(model)
            if population is None:
                population = optimizer.population_size
            elif optimizer.population_size != population:
                raise ValueError("all optimizer replicas must use the same population")
            self._replicas.append(
                _DeviceReplica(
                    device=device,
                    optimizer=optimizer,
                    model=model,
                    state=state,
                    candidate_ids=np.empty((0,), dtype=np.int32),
                )
            )

        assert population is not None
        self._population_size = population
        self._weights = list(weights or [1.0] * len(devices))
        self._partition_sizes = compute_partition_sizes(population, self._weights)
        offset = 0
        for replica, size in zip(self._replicas, self._partition_sizes, strict=True):
            replica.candidate_ids = np.arange(offset, offset + size, dtype=np.int32)
            offset += size

    def __enter__(self) -> "ReplicatedDistributedZeroGrad":
        return self

    def __exit__(self, *exc: object) -> bool:
        self.shutdown()
        return False

    def shutdown(self) -> None:
        executor = self._executor
        if executor is not None:
            executor.shutdown(wait=True)
            self._executor = None

    def _evaluate_replica(
        self,
        replica: _DeviceReplica,
        batch_host: Any,
        rng_host: Any,
        generation: int,
    ) -> np.ndarray:
        with jax.default_device(replica.device):
            batch = jax.device_put(batch_host, replica.device)
            candidate_ids = jax.device_put(replica.candidate_ids, replica.device)
            rng = jax.device_put(rng_host, replica.device)
            losses = replica.optimizer.evaluate_shard(
                replica.model,
                generation,
                self._loss_fn,
                batch,
                candidate_ids,
                rng=rng,
            )
            jax.block_until_ready(losses)
        return np.asarray(jax.device_get(losses), dtype=np.float32)

    def _update_replica(
        self,
        replica: _DeviceReplica,
        losses_host: np.ndarray,
    ) -> StepMetrics:
        with jax.default_device(replica.device):
            losses = jax.device_put(losses_host, replica.device)
            replica.model, replica.state, metrics = (
                replica.optimizer.step_from_losses(
                    replica.state,
                    replica.model,
                    losses,
                )
            )
            # A loss metric is independent of replay; wait on params to ensure
            # the deterministic update has completed before the next generation.
            jax.block_until_ready(params_pure_dict(replica.model))
        return metrics

    def step(
        self,
        batch: Any,
        *,
        rng: Array | None = None,
    ) -> tuple[Any, ZeroGradState, StepMetrics]:
        """Evaluate shards, exchange only losses, and replay on every GPU."""
        if self._executor is None:
            raise RuntimeError("coordinator has been shut down")
        if rng is None:
            rng = jax.random.key(0)

        generation = self._replicas[0].state.generation
        if any(replica.state.generation != generation for replica in self._replicas):
            raise RuntimeError("device replicas have diverged in generation")

        batch_host = jax.device_get(batch)
        rng_host = jax.device_get(rng)
        losses_host = gather_shard_losses(
            self._executor,
            self._replicas,
            lambda replica: self._evaluate_replica(replica, batch_host, rng_host, generation),
            concat=lambda parts: np.concatenate(parts, axis=0),
        )
        if losses_host.shape != (self._population_size,):
            raise RuntimeError(
                f"gathered losses have shape {losses_host.shape}, "
                f"expected {(self._population_size,)}"
            )

        update_futures = [
            self._executor.submit(self._update_replica, replica, losses_host)
            for replica in self._replicas
        ]
        metrics = [future.result() for future in update_futures]
        return self._replicas[0].model, self._replicas[0].state, metrics[0]

    @property
    def model(self) -> Any:
        return self._replicas[0].model

    @property
    def state(self) -> ZeroGradState:
        return self._replicas[0].state

    @property
    def devices(self) -> list[Device]:
        return list(self._devices)

    @property
    def partition_sizes(self) -> list[int]:
        return list(self._partition_sizes)

    @property
    def losses_bytes_per_step(self) -> int:
        # One gather plus one full loss-vector broadcast per replica.
        return self._population_size * np.dtype(np.float32).itemsize * (
            1 + len(self._replicas)
        )

    def verify_sync(self) -> bool:
        """Expensive debug check that all replicated parameter arrays match."""
        reference = [
            np.asarray(jax.device_get(value))
            for value in jax.tree.leaves(params_pure_dict(self._replicas[0].model))
        ]
        for replica in self._replicas[1:]:
            leaves = jax.tree.leaves(params_pure_dict(replica.model))
            if len(leaves) != len(reference):
                return False
            for expected, value in zip(reference, leaves, strict=True):
                if not np.array_equal(expected, np.asarray(jax.device_get(value))):
                    return False
        return True

    def restore_params(self, params: Any, *, generation: int) -> None:
        """Restore identical pure params on every replica (one-time checkpoint load)."""
        if generation < 0:
            raise ValueError("generation must be non-negative")
        for replica in self._replicas:
            if replica.state.opt_state is not None:
                raise ValueError(
                    "restore_params currently supports stateless/bin-update optimizers"
                )
            with jax.default_device(replica.device):
                params_device = jax.tree.map(
                    lambda value: jax.device_put(value, replica.device),
                    params,
                )
                update_params(replica.model, params_device)
                replica.state = ZeroGradState(generation=generation, opt_state=None)
                jax.block_until_ready(params_pure_dict(replica.model))