"""Tests for fault-tolerant cluster: late join, pause/resume, node death."""

import jax
import jax.numpy as jnp
import optax

from zerograd import (
    FaultTolerantCluster,
    ZeroGrad,
)


# ── Shared fixtures ───────────────────────────────────────────────────────────

from flax import nnx
from zerograd._nnx import params_pure_dict


class TinyMLP(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.l1 = nnx.Linear(4, 8, rngs=rngs)
        self.l2 = nnx.Linear(8, 2, rngs=rngs)

    def __call__(self, x):
        return self.l2(nnx.relu(self.l1(x)))


def build_model(key):
    return TinyMLP(nnx.Rngs(key))


def make_loss_fn():
    def loss_fn(model, batch):
        x, y = batch
        logits = model(x)
        return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, y)), None
    return loss_fn


def make_batch():
    return (
        jax.random.normal(jax.random.key(2), (16, 4)),
        jax.random.randint(jax.random.key(3), (16,), 0, 2),
    )


def make_optimizer(pop=16, seed=42, run_id="ft-test"):
    return ZeroGrad(
        optax.adamw(1e-2),
        population_size=pop,
        rank=4,
        sigma=0.1,
        seed=seed,
        run_id=run_id,
    )


def run_single_baseline(steps, seed=42, run_id="ft-test"):
    """Run single-node ZeroGrad for comparison."""
    opt = make_optimizer(seed=seed, run_id=run_id)
    model = build_model(jax.random.key(seed))
    state = opt.init(model)
    batch = make_batch()
    loss_fn = make_loss_fn()
    for _ in range(steps):
        model, state, _ = opt.step(state, model, batch, loss_fn)
    return model


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestFaultTolerantCluster:

    def test_basic_step_matches_single_node(self):
        """Cluster with 1 node produces identical results to single-node ZeroGrad."""
        opt = make_optimizer()
        loss_fn = make_loss_fn()
        batch = make_batch()

        fc = FaultTolerantCluster(opt, build_model, loss_fn, seed=42, initial_nodes=1)
        for _ in range(5):
            fc.step(batch)

        baseline = run_single_baseline(5)
        assert fc.verify_against_single(baseline, atol=1e-5)

    def test_multi_node_matches_single_node(self):
        """Cluster with 4 nodes matches single-node baseline."""
        opt = make_optimizer(pop=32)
        loss_fn = make_loss_fn()
        batch = make_batch()

        fc = FaultTolerantCluster(opt, build_model, loss_fn, seed=42, initial_nodes=4)
        for _ in range(10):
            fc.step(batch)

        assert fc.verify_sync()
        # Rebuild baseline with pop=32
        opt2 = make_optimizer(pop=32)
        params = build_model(jax.random.key(42))
        state = opt2.init(params)
        for _ in range(10):
            params, state, _ = opt2.step(state, params, batch, loss_fn)
        assert fc.verify_against_single(params, atol=1e-5)

    def test_late_join_catches_up(self):
        """A node added mid-training catches up via loss history replay."""
        opt = make_optimizer(pop=32)
        loss_fn = make_loss_fn()
        batch = make_batch()

        fc = FaultTolerantCluster(opt, build_model, loss_fn, seed=42, initial_nodes=2)

        # Run 10 steps
        for _ in range(10):
            fc.step(batch)

        # Add a late joiner
        idx = fc.add_node(name="late")
        assert fc.num_total_nodes == 3
        assert fc.verify_sync(), "late joiner should be synced after replay"

        # Late joiner's params should match
        # Rebuild with pop=32
        opt2 = make_optimizer(pop=32)
        params = build_model(jax.random.key(42))
        state = opt2.init(params)
        for _ in range(10):
            params, state, _ = opt2.step(state, params, batch, loss_fn)
        late_params = fc.nodes[idx].params
        for a, b in zip(
            jax.tree_util.tree_leaves(params_pure_dict(params)),
            jax.tree_util.tree_leaves(params_pure_dict(late_params)),
        ):
            assert float(jnp.max(jnp.abs(a - b))) < 1e-5

    def test_late_join_then_continue(self):
        """Late joiner stays in sync after continued training."""
        opt = make_optimizer(pop=32)
        loss_fn = make_loss_fn()
        batch = make_batch()

        fc = FaultTolerantCluster(opt, build_model, loss_fn, seed=42, initial_nodes=2)
        for _ in range(5):
            fc.step(batch)

        fc.add_node()
        assert fc.verify_sync()

        # Continue training with 3 nodes
        for _ in range(5):
            fc.step(batch)
        assert fc.verify_sync()

    def test_pause_resume(self):
        """Paused node stops evaluating, resumes with replay, stays synced."""
        opt = make_optimizer(pop=32)
        loss_fn = make_loss_fn()
        batch = make_batch()

        fc = FaultTolerantCluster(opt, build_model, loss_fn, seed=42, initial_nodes=3)
        for _ in range(5):
            fc.step(batch)

        # Pause node 1
        fc.pause_node(1)
        assert fc.num_active_nodes == 2
        assert fc.get_status(1).paused

        # Continue training with 2 active nodes
        for _ in range(5):
            fc.step(batch)

        # Resume node 1
        fc.resume_node(1)
        assert fc.num_active_nodes == 3
        assert not fc.get_status(1).paused
        assert fc.verify_sync(), "resumed node should be synced after replay"

    def test_node_death_redistributes_work(self):
        """Removing a node redistributes its work to survivors."""
        opt = make_optimizer(pop=32)
        loss_fn = make_loss_fn()
        batch = make_batch()

        fc = FaultTolerantCluster(opt, build_model, loss_fn, seed=42, initial_nodes=4)
        for _ in range(5):
            fc.step(batch)

        # Kill node 2
        fc.remove_node(2)
        assert fc.num_total_nodes == 3
        assert fc.num_active_nodes == 3
        assert sum(fc.partition_sizes) == 32, "all candidates must be covered"

        # Continue training
        for _ in range(5):
            fc.step(batch)
        assert fc.verify_sync()

    def test_weighted_partition_after_removal(self):
        """Weights are respected after node removal."""
        opt = make_optimizer(pop=32)
        loss_fn = make_loss_fn()

        fc = FaultTolerantCluster(opt, build_model, loss_fn, seed=42, initial_nodes=3)
        fc.set_weight(0, 1.0)
        fc.set_weight(1, 3.0)
        fc.set_weight(2, 1.0)

        sizes = fc.partition_sizes
        assert sizes[1] > sizes[0], "node 1 should have more candidates (weight=3)"

        # Remove weighted node
        fc.remove_node(1)
        sizes = fc.partition_sizes
        assert len(sizes) == 2
        assert sum(sizes) == 32

    def test_multiple_churn_events(self):
        """Simulate chaotic churn: adds, removes, pauses, resumes."""
        opt = make_optimizer(pop=32)
        loss_fn = make_loss_fn()
        batch = make_batch()

        fc = FaultTolerantCluster(opt, build_model, loss_fn, seed=42, initial_nodes=2)

        # Step 0-4: normal
        for _ in range(5):
            fc.step(batch)

        # Node 2 joins late
        fc.add_node(weight=2.0)
        assert fc.verify_sync()

        # Step 5-9: 3 nodes
        for _ in range(5):
            fc.step(batch)

        # Pause node 0
        fc.pause_node(0)
        for _ in range(5):
            fc.step(batch)

        # Kill node 1
        fc.remove_node(1)
        for _ in range(5):
            fc.step(batch)

        # Resume node 0
        fc.resume_node(0)
        assert fc.verify_sync()

        # Add another node
        fc.add_node()
        assert fc.verify_sync()

        # Final steps
        for _ in range(5):
            fc.step(batch)
        assert fc.verify_sync()

        # Verify against baseline (25 total steps)
        opt2 = make_optimizer(pop=32)
        params = build_model(jax.random.key(42))
        state = opt2.init(params)
        for _ in range(25):
            params, state, _ = opt2.step(state, params, batch, loss_fn)
        assert fc.verify_against_single(params, atol=1e-5)

    def test_loss_history_size(self):
        """Loss history grows with steps and is bounded by max_loss_history."""
        opt = make_optimizer(pop=16)
        loss_fn = make_loss_fn()
        batch = make_batch()

        fc = FaultTolerantCluster(
            opt, build_model, loss_fn, seed=42, initial_nodes=1,
            max_loss_history=5,
        )
        for _ in range(10):
            fc.step(batch)

        assert fc.loss_history_size == 5, "history should be capped at 5"

    def test_all_nodes_die_then_one_joins(self):
        """Edge case: remove all nodes, then add one — it replays from history."""
        opt = make_optimizer(pop=16)
        loss_fn = make_loss_fn()
        batch = make_batch()

        fc = FaultTolerantCluster(opt, build_model, loss_fn, seed=42, initial_nodes=2)
        for _ in range(5):
            fc.step(batch)

        fc.remove_node(1)
        fc.remove_node(0)
        assert fc.num_total_nodes == 0

        # Add a fresh node — it should catch up from loss history
        fc.add_node()
        assert fc.num_active_nodes == 1
        assert fc.verify_sync()  # trivially true with 1 node

        # Verify it matches baseline
        opt2 = make_optimizer(pop=16)
        params = build_model(jax.random.key(42))
        state = opt2.init(params)
        for _ in range(5):
            params, state, _ = opt2.step(state, params, batch, loss_fn)
        assert fc.verify_against_single(params, atol=1e-5)
