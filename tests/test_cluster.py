"""Tests for seed-derived cluster optimization: ZeroGradNode and ClusterZeroGrad."""

import jax
import jax.numpy as jnp
import pytest

from _tiny import batch as _batch, build_model as _build_model, make_opt, mse_loss as _loss_fn
from zerograd import (
    ClusterZeroGrad,
    ZeroGradNode,
)
from zerograd._nnx import params_pure_dict


def _make_opt(pop=8, **kw):
    return make_opt(pop=pop, run_id="cluster-test", **kw)


# ── ZeroGradNode ─────────────────────────────────────────────────────────────

class TestZeroGradNode:
    def test_seed_derives_params_and_state(self):
        opt = _make_opt()
        node = ZeroGradNode(opt, _build_model, _loss_fn, seed=42)
        # Same seed → identical params across nodes.
        other = ZeroGradNode(opt, _build_model, _loss_fn, seed=42)
        assert jnp.allclose(
            params_pure_dict(node.params)["l"]["kernel"],
            params_pure_dict(other.params)["l"]["kernel"],
        )
        assert node.state.generation == 0
        assert node.generation == 0
        assert node.seed == 42

    def test_evaluate_returns_losses_for_assigned_ids(self):
        opt = _make_opt()
        node = ZeroGradNode(opt, _build_model, _loss_fn, seed=42)
        ids = jnp.arange(4, dtype=jnp.int32)
        losses = node.evaluate(_batch(), ids)
        assert losses.shape == (4,)
        assert bool(jnp.all(jnp.isfinite(losses)))

    def test_step_advances_generation_and_returns_metrics(self):
        opt = _make_opt(pop=4)
        node = ZeroGradNode(opt, _build_model, _loss_fn, seed=42)
        losses = node.evaluate(_batch(), jnp.arange(4, dtype=jnp.int32))
        metrics = node.step(losses)
        assert node.generation == 1
        assert metrics.population_size == 4

    def test_step_does_not_share_params_with_coordinator(self):
        # The node updates its own params locally from losses.
        opt = _make_opt(pop=4)
        node = ZeroGradNode(opt, _build_model, _loss_fn, seed=42)
        before = jnp.array(params_pure_dict(node.params)["l"]["kernel"])
        losses = node.evaluate(_batch(), jnp.arange(4, dtype=jnp.int32))
        node.step(losses)
        # Params should change (loss is sum of squares → gradient nonzero).
        assert not jnp.allclose(before, params_pure_dict(node.params)["l"]["kernel"])


# ── ClusterZeroGrad ─────────────────────────────────────────────────────────

class TestClusterConstruction:
    def test_rejects_zero_nodes(self):
        with pytest.raises(ValueError):
            ClusterZeroGrad(_make_opt(), _build_model, _loss_fn, seed=42, num_nodes=0)

    def test_rejects_weights_longer_than_num_nodes(self):
        # Regression for issue #4: mismatched weights length must fail at
        # construction, not surface as a confusing losses-length error in step().
        with pytest.raises(ValueError):
            ClusterZeroGrad(
                _make_opt(), _build_model, _loss_fn, seed=42,
                num_nodes=2, weights=[1.0, 1.0, 1.0],
            )

    def test_rejects_weights_shorter_than_num_nodes(self):
        with pytest.raises(ValueError):
            ClusterZeroGrad(
                _make_opt(), _build_model, _loss_fn, seed=42,
                num_nodes=3, weights=[1.0],
            )

    def test_default_equal_weights_partition(self):
        cluster = ClusterZeroGrad(
            _make_opt(pop=8), _build_model, _loss_fn, seed=42, num_nodes=4,
        )
        assert cluster.partition_sizes == [2, 2, 2, 2]
        assert cluster.weights == [1.0, 1.0, 1.0, 1.0]

    def test_weighted_partition(self):
        cluster = ClusterZeroGrad(
            _make_opt(pop=10), _build_model, _loss_fn, seed=42,
            num_nodes=2, weights=[1.0, 4.0],
        )
        assert cluster.partition_sizes == [2, 8]

    def test_nodes_all_seed_derived_identical(self):
        cluster = ClusterZeroGrad(
            _make_opt(), _build_model, _loss_fn, seed=42, num_nodes=3,
        )
        leaves0 = jax.tree_util.tree_leaves(cluster.nodes[0].params)
        for node in cluster.nodes[1:]:
            for a, b in zip(leaves0, jax.tree_util.tree_leaves(node.params)):
                assert jnp.array_equal(a, b)
        # verify_sync trivially true at generation 0
        assert cluster.verify_sync()

    def test_communication_accounting(self):
        pop = 8
        cluster = ClusterZeroGrad(
            _make_opt(pop=pop), _build_model, _loss_fn, seed=42, num_nodes=2,
        )
        # losses_bytes = pop * 4 (float32)
        assert cluster.losses_bytes_per_step == pop * 4
        # params_bytes = param_count * 4 (numeric param leaves after surgery)
        param_count = sum(
            v.size for v in jax.tree_util.tree_leaves(params_pure_dict(cluster.nodes[0].params))
        )
        assert cluster.params_bytes == param_count * 4

    def test_init_returns_node_zero_state(self):
        cluster = ClusterZeroGrad(_make_opt(), _build_model, _loss_fn, seed=42, num_nodes=2)
        state = cluster.init()
        assert state.generation == 0
        assert state is cluster.nodes[0].state


class TestClusterStep:
    def test_step_advances_all_nodes_and_syncs(self):
        cluster = ClusterZeroGrad(
            _make_opt(pop=8), _build_model, _loss_fn, seed=42, num_nodes=3,
        )
        params, state, metrics = cluster.step(_batch())
        assert state.generation == 1
        assert metrics.population_size == 8
        assert cluster.verify_sync(), "all nodes must converge to identical params"

    def test_single_node_cluster_matches_plain_optimizer(self):
        opt = _make_opt(pop=8)
        params = _build_model(jax.random.key(42))
        state = opt.init(params)
        batch = _batch()
        ref_params, _, _ = opt.step(state, params, batch, _loss_fn)

        cluster = ClusterZeroGrad(opt, _build_model, _loss_fn, seed=42, num_nodes=1)
        cparams, cstate, _ = cluster.step(batch)
        assert cstate.generation == 1
        # Single-node cluster: verify_sync short-circuits to True.
        assert cluster.verify_sync()
        for a, b in zip(jax.tree_util.tree_leaves(cparams), jax.tree_util.tree_leaves(ref_params)):
            assert jnp.allclose(a, b)

    def test_multi_node_matches_single_node_baseline(self):
        opt = _make_opt(pop=16)
        batch = _batch()

        # Baseline: single optimizer
        params = _build_model(jax.random.key(42))
        state = opt.init(params)
        for _ in range(3):
            params, state, _ = opt.step(state, params, batch, _loss_fn)

        # Cluster with 4 nodes must match after the same number of steps.
        cluster = ClusterZeroGrad(opt, _build_model, _loss_fn, seed=42, num_nodes=4)
        for _ in range(3):
            cparams, cstate, _ = cluster.step(batch)
        assert cluster.verify_sync()
        for a, b in zip(jax.tree_util.tree_leaves(cparams), jax.tree_util.tree_leaves(params)):
            assert jnp.allclose(a, b, atol=1e-5)

    def test_weighted_cluster_still_syncs(self):
        cluster = ClusterZeroGrad(
            _make_opt(pop=16), _build_model, _loss_fn, seed=42,
            num_nodes=3, weights=[1.0, 2.0, 1.0],
        )
        assert cluster.partition_sizes == [4, 8, 4]
        cluster.step(_batch())
        assert cluster.verify_sync()

    def test_verify_sync_detects_drift_when_desynced(self):
        # Manually perturb one node to confirm verify_sync is a real check.
        cluster = ClusterZeroGrad(
            _make_opt(), _build_model, _loss_fn, seed=42, num_nodes=2,
        )
        # Overwrite node 1's params with different values.
        cluster.nodes[1]._params = {"w": jnp.ones((4, 2)), "v": jnp.zeros((2,))}
        assert not cluster.verify_sync()
