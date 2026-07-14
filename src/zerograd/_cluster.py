"""Seed-derived cluster optimization for ZeroGrad.

Each node computes its own parameters from a shared seed and the sequence
of fitness arrays. Only the 1D loss array (O(population) bytes) is
communicated between nodes — no params, gradients, or optimizer state
ever cross node boundaries.

This enables scaling to hundreds of nodes with minimal network overhead:
the per-step communication cost is independent of model size.

Protocol
--------
1. Init: coordinator sends seed (4 bytes). Each node computes
   ``params_0 = build_params(seed)`` and ``state_0 = optimizer.init(params_0)``
   locally.

2. Each step:
   a. Each node evaluates its candidate shard (forward passes only).
   b. Nodes send loss arrays to coordinator — O(pop/nodes) floats each.
   c. Coordinator gathers and broadcasts the full loss array — O(pop) floats.
   d. Each node independently calls ``step_from_losses`` — all arrive at
      identical params because the update is deterministic from
      (params, losses, seed-derived factors).

3. The params are never on the network. They are a pure function of
   (seed, loss_history) — any node can recompute them locally.

Why it works
------------
- Params are seed-derived: ``build_params(seed)`` is deterministic.
- Candidate perturbations are seed-derived: factor keys come from
  ``(seed, run_id, generation, candidate_id)``.
- The pseudo-gradient replay uses the same keys → same factors → same update.
- Optax state evolves deterministically from the same pseudo-gradient sequence.

Communication cost
------------------
Per step: O(population) bytes for losses, regardless of model size.
A 1B-parameter model and a 1M-parameter model have the same per-step
network cost. Compare to backprop's O(parameters) gradient sync.
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp

from ._distributed import compute_partition_sizes
from ._manifest import ParameterTree
from ._optimizer import LossFn, StepMetrics, ZeroGrad, ZeroGradState

Array = jax.Array
ParamsBuilder = Callable[[Array], ParameterTree]


class ZeroGradNode:
    """One independent node in a seed-derived cluster.

    Parameters and optimizer state are derived deterministically from a
    seed. The node never receives params — it computes them locally from
    the seed and the sequence of fitness arrays it receives.

    In a multi-process cluster, each node runs in its own process. The
    only data that crosses process boundaries is the 1D loss array.

    Parameters:
        optimizer: The ZeroGrad optimizer (provides evaluate_shard and
            step_from_losses).
        build_params_fn: Deterministic function from PRNG key to parameter
            tree. Must be identical across all nodes.
        loss_fn: The loss function for candidate evaluation.
        seed: Integer seed. All nodes with the same seed start from the
            same params and stay in sync.
    """

    def __init__(
        self,
        optimizer: ZeroGrad,
        build_params_fn: ParamsBuilder,
        loss_fn: LossFn,
        seed: int,
    ) -> None:
        self._optimizer = optimizer
        self._loss_fn = loss_fn
        self._seed = seed
        # Compute params locally from seed — never received from coordinator
        self._params = build_params_fn(jax.random.key(seed))
        self._state = optimizer.init(self._params)

    def evaluate(self, batch: Any, candidate_ids: Array) -> Array:
        """Evaluate assigned candidates and return their losses.

        This is the only computation whose result leaves the node.
        The loss array is O(len(candidate_ids)) floats — a few hundred
        bytes at most.
        """
        gen = self._state.generation
        return self._optimizer.evaluate_shard(
            self._params, gen, self._loss_fn, batch, candidate_ids)

    def step(self, losses: Array) -> StepMetrics:
        """Apply update from gathered losses.

        Params are updated locally. Because the update is deterministic
        from (params, losses, seed-derived factors), every node that
        receives the same loss array arrives at identical params.
        """
        self._params, self._state, metrics = self._optimizer.step_from_losses(
            self._state, self._params, losses)
        return metrics

    @property
    def params(self) -> ParameterTree:
        """Current params (computed locally from seed + loss history)."""
        return self._params

    @property
    def state(self) -> ZeroGradState:
        return self._state

    @property
    def generation(self) -> int:
        return self._state.generation

    @property
    def seed(self) -> int:
        return self._seed


class ClusterZeroGrad:
    """Multi-node coordinator that shares only fitness arrays.

    Each node independently maintains its own parameter tree from a shared
    seed. Per step:

    1. Each node evaluates its candidate shard (forward passes only).
    2. Losses are gathered — this is the ONLY data crossing node boundaries.
    3. The full loss array is broadcast to all nodes.
    4. Each node independently calls ``step_from_losses`` — all arrive at
       identical params.

    Communication per step: O(population) bytes, independent of model size.

    Parameters:
        optimizer: The ZeroGrad optimizer to coordinate.
        build_params_fn: Deterministic function from PRNG key to params.
            Must be the same across all nodes.
        loss_fn: The loss function for candidate evaluation.
        seed: Integer seed shared by all nodes.
        num_nodes: Number of independent nodes in the cluster.
        weights: Relative compute weights, one per node. ``None`` means
            equal weights (even split). Example: ``[1, 4]`` gives node 1
            four times as many candidates as node 0.
    """

    def __init__(
        self,
        optimizer: ZeroGrad,
        build_params_fn: ParamsBuilder,
        loss_fn: LossFn,
        seed: int,
        num_nodes: int = 1,
        weights: list[float] | None = None,
    ) -> None:
        if num_nodes < 1:
            raise ValueError("num_nodes must be >= 1")
        self._optimizer = optimizer
        self._seed = seed
        self._nodes: list[ZeroGradNode] = [
            ZeroGradNode(optimizer, build_params_fn, loss_fn, seed)
            for _ in range(num_nodes)
        ]

        pop = optimizer.population_size
        w = weights or [1.0] * num_nodes
        self._weights = w
        sizes = compute_partition_sizes(pop, w)
        all_ids = jnp.arange(pop, dtype=jnp.int32)
        if len(sizes) > 1:
            split_points = jnp.cumsum(jnp.array(sizes[:-1]))
            self._shard_ids = jnp.split(all_ids, split_points)
        else:
            self._shard_ids = [all_ids]
        self._partition_sizes = sizes

        # Communication accounting
        self._losses_bytes_per_step = pop * 4  # float32
        param_count = sum(
            v.size for v in jax.tree_util.tree_leaves(self._nodes[0].params)
        )
        self._params_bytes = param_count * 4  # float32

    @property
    def nodes(self) -> list[ZeroGradNode]:
        """The independent nodes in this cluster."""
        return self._nodes

    @property
    def partition_sizes(self) -> list[int]:
        """Number of candidates assigned to each node."""
        return list(self._partition_sizes)

    @property
    def weights(self) -> list[float]:
        """Current compute weights, one per node."""
        return list(self._weights)

    @property
    def losses_bytes_per_step(self) -> int:
        """Bytes of loss data communicated per step (model-size independent)."""
        return self._losses_bytes_per_step

    @property
    def params_bytes(self) -> int:
        """Bytes of parameter data that would be needed for param sync."""
        return self._params_bytes

    def init(self) -> ZeroGradState:
        """Nodes are already initialized from seed. Return node 0's state."""
        return self._nodes[0].state

    def step(self, batch: Any) -> tuple[ParameterTree, ZeroGradState, StepMetrics]:
        """Execute one cluster step — only fitness arrays are shared.

        1. Each node evaluates its candidate shard.
        2. Losses are gathered (ONLY data crossing node boundaries).
        3. Each node independently calls step_from_losses.
        4. All nodes arrive at identical params.
        """
        # 1. Each node evaluates its shard
        all_losses = []
        for node, ids in zip(self._nodes, self._shard_ids):
            losses = node.evaluate(batch, ids)
            all_losses.append(losses)

        # 2. Gather losses — ONLY data crossing node boundaries
        gathered = jnp.concatenate(all_losses)

        # 3. Each node independently computes the same update
        metrics = None
        for node in self._nodes:
            metrics = node.step(gathered)

        return self._nodes[0].params, self._nodes[0].state, metrics

    def verify_sync(self, atol: float = 1e-5) -> bool:
        """Verify all nodes have identical params.

        Proves that no param communication was needed — every node
        independently arrived at the same parameter tree using only
        the shared seed and loss arrays.
        """
        if len(self._nodes) < 2:
            return True
        ref_leaves = jax.tree_util.tree_leaves(self._nodes[0].params)
        for node in self._nodes[1:]:
            node_leaves = jax.tree_util.tree_leaves(node.params)
            for a, b in zip(ref_leaves, node_leaves):
                if float(jnp.max(jnp.abs(a - b))) > atol:
                    return False
        return True
