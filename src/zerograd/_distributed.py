"""Multi-device distributed evaluation for ZeroGrad.

The ES population is embarrassingly parallel: each candidate's loss is
computed independently, and only the 1D fitness array needs to be shared
between workers.  This module provides:

- ``DeviceShard``: evaluates a subset of candidates on one JAX device.
- ``DistributedZeroGrad``: coordinates multiple shards, gathers fitnesses,
  and drives the optimizer step.

Workers can be mixed across CPU and GPU, or multiple shards can share a
single GPU.  Each worker compiles its own loss function for its device.
"""

from __future__ import annotations

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


class DeviceShard:
    """Evaluate a population shard on a specific JAX device.

    Each shard JIT-compiles its evaluation function for its assigned device.
    The compiled function is reused across all generations.

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
        """Evaluate the assigned candidates on this shard's device.

        Params and batch are placed on the device, evaluated, and the
        resulting losses are returned (still on-device — caller gathers).
        """
        if rng is None:
            rng = jax.random.key(0)

        # Place data on this device
        params_d = jax.device_put(params, self.device)
        batch_d = jax.device_put(batch, self.device)
        ids_d = jax.device_put(candidate_ids, self.device)
        rng_d = jax.device_put(rng, self.device)

        losses = self._optimizer.evaluate_shard(
            params_d, generation, self._loss_fn, batch_d, ids_d, rng=rng_d,
        )
        return ShardResult(candidate_ids=candidate_ids, losses=losses)


class DistributedZeroGrad:
    """Multi-device ZeroGrad coordinator.

    Splits the population into shards, one per device.  Each step:
    1. Splits candidate IDs evenly across devices.
    2. Evaluates each shard on its device (concurrently via threads).
    3. Gathers all losses and sorts them into candidate-ID order.
    4. Calls ``step_from_losses`` to complete the generation.

    Only the 1D loss arrays cross device boundaries — typically a few
    hundred floats per step, negligible overhead.

    Parameters:
        optimizer: The ZeroGrad optimizer to coordinate.
        devices: List of JAX devices to evaluate on.
        loss_fn: The loss function for candidate evaluation.
        coordinator_device: Device for the step_from_losses computation
            (defaults to the first device).
    """

    def __init__(
        self,
        optimizer: ZeroGrad,
        devices: list[jax.Device],
        loss_fn: LossFn,
        coordinator_device: jax.Device | None = None,
    ) -> None:
        if not devices:
            raise ValueError("at least one device is required")
        pop = optimizer.population_size
        if pop % len(devices) != 0:
            raise ValueError(
                f"population_size ({pop}) must be divisible by number of "
                f"devices ({len(devices)})"
            )
        self._optimizer = optimizer
        self._devices = devices
        self._coordinator_device = coordinator_device or devices[0]

        # Create one shard per device
        self._shards: list[DeviceShard] = [
            DeviceShard(dev, optimizer, loss_fn, name=f"worker-{i}-{dev.platform}")
            for i, dev in enumerate(devices)
        ]

        # Pre-compute candidate-ID splits
        all_ids = jnp.arange(optimizer.population_size, dtype=jnp.int32)
        self._shard_ids = jnp.split(all_ids, len(devices))

        # Thread pool for concurrent evaluation
        self._executor = ThreadPoolExecutor(max_workers=len(devices))

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

        Splits the population across devices, evaluates concurrently,
        gathers losses, and completes the step on the coordinator device.
        """
        if rng is None:
            rng = jax.random.key(0)

        gen = state.generation

        # Dispatch evaluation to each shard concurrently
        futures = []
        for shard, ids in zip(self._shards, self._shard_ids):
            future = self._executor.submit(
                shard.evaluate, params, batch, ids, gen, rng,
            )
            futures.append(future)

        # Gather results
        results: list[ShardResult] = [f.result() for f in futures]

        # Sort losses into candidate-ID order on the coordinator device
        all_losses = []
        for result in results:
            losses = jax.device_put(result.losses, self._coordinator_device)
            all_losses.append(losses)
        losses = jnp.concatenate(all_losses)

        # Complete the step on the coordinator device
        params_c = jax.device_put(params, self._coordinator_device)
        return self._optimizer.step_from_losses(state, params_c, losses)

    @property
    def shards(self) -> list[DeviceShard]:
        """The device shards managed by this coordinator."""
        return self._shards
