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
from typing import Any

import jax
import jax.numpy as jnp

from ._manifest import ParameterTree
from ._optimizer import LossFn, StepMetrics, ZeroGrad, ZeroGradState

Array = jax.Array


@dataclass(frozen=True, slots=True)
class ShardResult:
    """One worker's contribution: candidate IDs and their losses."""

    candidate_ids: Array
    losses: Array


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    """Per-device timing from auto-calibration."""

    device: jax.Device
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
        device: jax.Device,
        optimizer: ZeroGrad,
        loss_fn: LossFn,
        name: str = "",
    ) -> None:
        self.device = device
        self._optimizer = optimizer
        self._loss_fn = loss_fn
        self.name = name or f"shard-{device.platform}"

    def evaluate(
        self,
        params: ParameterTree,
        batch: Any,
        candidate_ids: Array,
        generation: int,
        rng: Array | None = None,
    ) -> ShardResult:
        """Evaluate the assigned candidates on this shard's device."""
        if rng is None:
            rng = jax.random.key(0)

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
        devices: list[jax.Device],
        loss_fn: LossFn,
        weights: list[float] | None = None,
        coordinator_device: jax.Device | None = None,
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
        all_ids = jnp.arange(self._optimizer.population_size, dtype=jnp.int32)
        sizes = self._partition_sizes

        # Build split points (cumulative sum, excluding last element)
        split_points = jnp.cumsum(jnp.array(sizes[:-1])) if len(sizes) > 1 else []
        id_shards = jnp.split(all_ids, split_points) if len(sizes) > 1 else [all_ids]

        for shard, ids in zip(self._shards, id_shards):
            shard._candidate_ids = ids

    def init(self, params: ParameterTree) -> ZeroGradState:
        """Initialize optimizer state."""
        return self._optimizer.init(params)

    def step(
        self,
        state: ZeroGradState,
        params: ParameterTree,
        batch: Any,
        *,
        rng: Array | None = None,
    ) -> tuple[ParameterTree, ZeroGradState, StepMetrics]:
        """Execute one distributed optimization generation.

        Evaluates each shard concurrently, gathers losses in candidate-ID
        order, and completes the step on the coordinator device.
        """
        if rng is None:
            rng = jax.random.key(0)
        if self._executor is None:
            raise RuntimeError("coordinator has been shut down")

        gen = state.generation

        # Dispatch evaluation to each shard concurrently
        futures = []
        for shard in self._shards:
            ids = shard._candidate_ids
            if ids is not None and ids.shape[0] > 0:
                future = self._executor.submit(
                    shard.evaluate, params, batch, ids, gen, rng,
                )
            else:
                future = None
            futures.append(future)

        # Gather results in shard order (shards already hold sequential candidate ID ranges)
        all_losses = []
        for future in futures:
            if future is None:
                continue
            result = future.result()
            losses = jax.device_put(result.losses, self._coordinator_device)
            all_losses.append(losses)
        losses = jnp.concatenate(all_losses)

        # Complete the step on the coordinator device
        params_c = jax.device_put(params, self._coordinator_device)
        return self._optimizer.step_from_losses(state, params_c, losses)

    def calibrate(
        self,
        params: ParameterTree,
        batch: Any,
        *,
        warmup: int = 1,
        trials: int = 3,
        rng: Array | None = None,
    ) -> list[CalibrationResult]:
        """Auto-calibrate weights by timing each device.

        Runs ``warmup`` evaluation rounds (to trigger JIT compilation) then
        ``trials`` timed rounds.  Weights are set inversely proportional to
        per-candidate evaluation time — a device that is 4× faster gets 4×
        more candidates.

        After calibration, the partition is immediately updated.  Call this
        once before training starts, after ``init``.

        Parameters:
            params: Sample parameter tree.
            batch: Sample batch.
            warmup: Number of untimed warmup rounds (for JIT compilation).
            trials: Number of timed rounds to average.
            rng: Optional RNG key.
        """
        if rng is None:
            rng = jax.random.key(0)

        pop = self._optimizer.population_size
        # Use a small fixed subset of candidates for calibration timing
        calib_ids = jnp.arange(min(pop, 4), dtype=jnp.int32)
        gen = 0

        results: list[CalibrationResult] = []

        for shard in self._shards:
            dev = shard.device
            # Warmup (JIT compilation)
            for _ in range(warmup):
                r = shard.evaluate(params, batch, calib_ids, gen, rng)
                jax.block_until_ready(r.losses)

            # Timed runs
            t0 = time.perf_counter()
            for _ in range(trials):
                r = shard.evaluate(params, batch, calib_ids, gen, rng)
                jax.block_until_ready(r.losses)
            elapsed = (time.perf_counter() - t0) / trials
            per_candidate = elapsed / len(calib_ids)

            results.append(CalibrationResult(
                device=dev,
                name=shard.name,
                num_candidates=len(calib_ids),
                elapsed_seconds=elapsed,
                per_candidate_seconds=per_candidate,
            ))

        # Set weights inversely proportional to per-candidate time
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