"""Fault-tolerant seed-derived cluster for ZeroGrad.

Extends :class:`ClusterZeroGrad` with node lifecycle management:
late joining, pause/resume, and node death with work redistribution.

The key insight: since parameters are a pure function of
``(seed, loss_history)``, any node can catch up by replaying the loss
history. The coordinator stores a log of every generation's loss array.
A late-joining node replays from generation 0 to the current generation
and arrives at the same params as all other nodes.

Node states
-----------
- **active**: participates in evaluation and stepping
- **paused**: temporarily not evaluating; its candidate IDs are
  redistributed to active nodes. It keeps stepping (so it stays in
  sync) and resumes evaluating on demand.
- **dead**: permanently removed. Its candidate IDs are redistributed.

When nodes join/leave, the coordinator recomputes the partition across
active nodes. The candidate IDs are always 0..population-1 — only the
assignment of IDs to nodes changes. Since every node independently
computes the full ``step_from_losses`` (which uses all candidate IDs),
the partition only affects *who evaluates which candidates*, not the
mathematical result.

This is designed for unreliable, asymmetric, decentralized compute:
gaming PCs that go offline, spot instances that get preempted, nodes
that join mid-training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp

from ._cluster import ParamsBuilder, ZeroGradNode
from ._distributed import compute_partition_sizes
from ._manifest import ParameterTree
from ._optimizer import LossFn, StepMetrics, ZeroGrad, ZeroGradState

Array = jax.Array

#: Default loss-history retention window. Finite so long runs do not leak
#: memory unbounded by default (see issue #21); ``0`` opts into unlimited
#: retention for users who rely on late-joiner catch-up beyond this window.
DEFAULT_MAX_LOSS_HISTORY = 1024


@dataclass(slots=True)
class NodeStatus:
    """Lifecycle state of one node in the fault-tolerant cluster."""

    name: str
    node: ZeroGradNode
    weight: float = 1.0
    active: bool = True
    paused: bool = False
    last_generation: int = 0  # last generation this node has stepped to
    # Candidate IDs assigned to this node by the most recent repartition.
    # Declared as a field (rather than a dynamic attribute) so NodeStatus can
    # use slots matching the codebase value-type convention (see issue #23).
    _shard_ids: Array | None = field(default=None, init=False, repr=False)


class FaultTolerantCluster:
    """Cluster coordinator with node lifecycle management.

    Handles late joining, pause/resume, and node death. The coordinator
    maintains a loss history log so any node can catch up by replaying
    missed generations.

    Parameters:
        optimizer: The ZeroGrad optimizer.
        build_params_fn: Deterministic function from PRNG key to params.
        loss_fn: The loss function for candidate evaluation.
        seed: Integer seed shared by all nodes.
        initial_nodes: Number of nodes to create at init (default 1).
        max_loss_history: Maximum generations of loss history to retain.
            Older history is discarded. Defaults to a finite cap
            (``DEFAULT_MAX_LOSS_HISTORY``) so long runs do not leak memory;
            set to 0 for unlimited retention. Adding a node once truncation
            has dropped generation 0 raises ``RuntimeError``, since a fresh
            node cannot catch up.
    """

    def __init__(
        self,
        optimizer: ZeroGrad,
        build_params_fn: ParamsBuilder,
        loss_fn: LossFn,
        seed: int,
        initial_nodes: int = 1,
        max_loss_history: int = DEFAULT_MAX_LOSS_HISTORY,
    ) -> None:
        if not isinstance(initial_nodes, int) or isinstance(initial_nodes, bool) or initial_nodes < 1:
            raise ValueError("initial_nodes must be a positive integer")
        if not isinstance(max_loss_history, int) or isinstance(max_loss_history, bool) or max_loss_history < 0:
            raise ValueError("max_loss_history must be a non-negative integer")
        self._optimizer = optimizer
        self._build_params_fn = build_params_fn
        self._loss_fn = loss_fn
        self._seed = seed
        self._pop = optimizer.population_size
        self._max_loss_history = max_loss_history

        # Loss history: list of loss arrays, indexed by generation
        self._loss_history: list[Array] = []
        # Monotonic count of completed steps (independent of history retention)
        self._step_count = 0

        # Node registry
        self._statuses: list[NodeStatus] = []
        self._next_node_id = 0
        for _ in range(initial_nodes):
            self.add_node()

    # ── Node lifecycle ─────────────────────────────────────────────────────

    def add_node(self, weight: float = 1.0, name: str | None = None) -> int:
        """Add a new node to the cluster. Returns its index.

        The node computes its params from the seed, then replays the
        loss history to catch up to the current generation. Catch-up is
        only possible when the retained history reaches back to
        generation 0; if ``max_loss_history`` has truncated early
        generations, this raises ``RuntimeError`` rather than silently
        producing a divergent node.

        After adding, the partition is redistributed across active nodes.
        """
        node_name = name or f"node-{self._next_node_id}"
        node = ZeroGradNode(
            self._optimizer, self._build_params_fn, self._loss_fn, self._seed,
        )
        status = NodeStatus(
            name=node_name, node=node, weight=weight,
            last_generation=0,
        )

        # Catch up: replay loss history from generation 0 forward. The
        # node starts at generation 0, and step_from_losses derives its
        # replay keys from the node's internal generation, so the history
        # must reach back to generation 0 for the keys to line up.
        if self._loss_history:
            if self._step_count != len(self._loss_history):
                raise RuntimeError(
                    "cannot add a node: loss history has been truncated "
                    f"(retained {len(self._loss_history)} of "
                    f"{self._step_count} generations) and no longer reaches "
                    "back to generation 0; a new node cannot catch up"
                )
            for losses in self._loss_history:
                node.step(losses)
            status.last_generation = node.generation

        idx = len(self._statuses)
        self._statuses.append(status)
        self._next_node_id += 1
        self._repartition()
        return idx

    def remove_node(self, index: int) -> NodeStatus:
        """Remove a node (simulates death). Its work is redistributed.

        The node's candidate IDs are redistributed to surviving active
        nodes. No data is lost — the coordinator still has the full loss
        history.
        """
        status = self._statuses.pop(index)
        self._repartition()
        return status

    def pause_node(self, index: int) -> None:
        """Temporarily pause a node. Its work is redistributed.

        The node stops evaluating but keeps its params in sync by
        continuing to step with the gathered losses each generation.
        """
        self._statuses[index].paused = True
        self._statuses[index].active = False
        self._repartition()

    def resume_node(self, index: int) -> None:
        """Resume a paused node.

        Paused nodes continue stepping during training (they receive the
        gathered losses each generation), so the node is already caught up
        to the current generation — no replay is needed. The node resumes
        evaluating and its candidate IDs are re-integrated into the active
        partition.
        """
        status = self._statuses[index]
        status.paused = False
        status.active = True

        self._repartition()

    def set_weight(self, index: int, weight: float) -> None:
        """Update a node's compute weight and repartition."""
        self._statuses[index].weight = weight
        self._repartition()

    # ── Partitioning ───────────────────────────────────────────────────────

    def _repartition(self) -> None:
        """Recompute candidate ID assignment across active nodes."""
        active = [s for s in self._statuses if s.active]
        if not active:
            return

        weights = [s.weight for s in active]
        sizes = compute_partition_sizes(self._pop, weights)

        all_ids = jnp.arange(self._pop, dtype=jnp.int32)
        if len(sizes) > 1:
            split_points = jnp.cumsum(jnp.array(sizes[:-1]))
            id_shards = jnp.split(all_ids, split_points)
        else:
            id_shards = [all_ids]

        for status, ids in zip(active, id_shards):
            status._shard_ids = ids

    def _get_active_shards(self) -> list[tuple[NodeStatus, Array]]:
        """Return (status, candidate_ids) for all active nodes."""
        active = [s for s in self._statuses if s.active]
        return [(s, s._shard_ids) for s in active]

    @property
    def partition_sizes(self) -> list[int]:
        """Candidate counts per active node."""
        return [int(s._shard_ids.shape[0]) for s in self._statuses if s.active]

    @property
    def num_active_nodes(self) -> int:
        return sum(1 for s in self._statuses if s.active)

    @property
    def num_total_nodes(self) -> int:
        return len(self._statuses)

    @property
    def generation(self) -> int:
        """Current generation (number of completed steps)."""
        return self._step_count

    @property
    def max_loss_history(self) -> int:
        """Configured loss-history retention cap (0 means unlimited)."""
        return self._max_loss_history

    @property
    def loss_history_size(self) -> int:
        """Number of generations of loss history retained."""
        return len(self._loss_history)

    @property
    def losses_bytes_per_step(self) -> int:
        return self._pop * 4

    @property
    def params_bytes(self) -> int:
        param_count = sum(
            v.size for v in jax.tree_util.tree_leaves(
                self._statuses[0].node.params
            )
        ) if self._statuses else 0
        return param_count * 4

    @property
    def nodes(self) -> list[ZeroGradNode]:
        """All nodes (including paused)."""
        return [s.node for s in self._statuses]

    @property
    def active_nodes(self) -> list[ZeroGradNode]:
        """Active nodes only."""
        return [s.node for s in self._statuses if s.active]

    def get_status(self, index: int) -> NodeStatus:
        """Get lifecycle status of a node."""
        return self._statuses[index]

    def init(self) -> ZeroGradState:
        """Return node 0's state (all nodes start identical from seed)."""
        if not self._statuses:
            raise RuntimeError("no nodes available; cannot init")
        return self._statuses[0].node.state

    # ── Training step ──────────────────────────────────────────────────────

    def step(self, batch: Any) -> tuple[ParameterTree, ZeroGradState, StepMetrics]:
        """Execute one generation across active nodes.

        1. Active nodes evaluate their candidate shards.
        2. Losses are gathered (ONLY data crossing node boundaries).
        3. Full loss array is stored in history and broadcast to all nodes.
        4. All nodes (active and paused that are catching up) step.

        Paused nodes do not evaluate but do step (they'll catch up on
        resume). This keeps them in sync if they're still running.
        """
        active_shards = self._get_active_shards()
        if not active_shards:
            raise RuntimeError("no active nodes available; cannot step")

        # 1. Each active node evaluates its shard
        all_losses = []
        for status, ids in active_shards:
            losses = status.node.evaluate(batch, ids)
            all_losses.append(losses)

        # 2. Gather losses
        gathered = jnp.concatenate(all_losses)

        # 3. Store in loss history
        self._loss_history.append(gathered)
        if self._max_loss_history > 0 and len(self._loss_history) > self._max_loss_history:
            self._loss_history = self._loss_history[-self._max_loss_history:]
        self._step_count += 1

        # 4. All nodes step (active + paused ones that are still alive)
        metrics = None
        for status in self._statuses:
            if status.active or status.paused:
                metrics = status.node.step(gathered)
                status.last_generation = status.node.generation

        # Return from first active node
        ref = self._statuses[0]
        return ref.node.params, ref.node.state, metrics

    # ── Verification ───────────────────────────────────────────────────────

    def verify_sync(self, atol: float = 1e-5) -> bool:
        """Verify all nodes (active and paused) have identical params."""
        if len(self._statuses) < 2:
            return True
        ref_leaves = jax.tree_util.tree_leaves(self._statuses[0].node.params)
        for status in self._statuses[1:]:
            node_leaves = jax.tree_util.tree_leaves(status.node.params)
            for a, b in zip(ref_leaves, node_leaves):
                if float(jnp.max(jnp.abs(a - b))) > atol:
                    return False
        return True

    def verify_against_single(
        self, params: ParameterTree, atol: float = 1e-5,
    ) -> bool:
        """Verify cluster params match an externally-computed baseline."""
        ref_leaves = jax.tree_util.tree_leaves(params)
        cluster_leaves = jax.tree_util.tree_leaves(self._statuses[0].node.params)
        for a, b in zip(ref_leaves, cluster_leaves):
            if float(jnp.max(jnp.abs(a - b))) > atol:
                return False
        return True
