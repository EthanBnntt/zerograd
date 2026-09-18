"""Regression tests for library issues #14-#26.

Each test pins the behaviour fixed in the corresponding GitHub issue so the
fix cannot silently regress.
"""

from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from _tiny import Tiny, make_opt
from _tiny import batch as _batch
from _tiny import build_model as _build_model
from _tiny import mse_loss as _loss_fn
from flax import nnx

from zerograd import (
    ClusterZeroGrad,
    FaultTolerantCluster,
    Manifest,
    ManifestEntry,
    NodeStatus,
    ParameterLayout,
    ZeroGrad,
    compute_partition_sizes,
    shape_centered_loss,
    validate_losses,
)
from zerograd._factors import matrix_factors
from zerograd._fault_tolerant import DEFAULT_MAX_LOSS_HISTORY


def _manifest():
    return Manifest(version=1, entries=(
        ManifestEntry(("w",), ParameterLayout.MATRIX, "w"),
        ManifestEntry(("v",), ParameterLayout.VECTOR, "v"),
    ))


def _make_opt(**kw):
    return make_opt(run_id="regress", **kw)


def _trivial_loss_fn(model, batch):
    return jnp.sum(model.l.kernel[...] ** 2), None


class TestIssue14IntegerLosses:
    def test_shape_centered_loss_rejects_integer(self):
        with pytest.raises(TypeError):
            shape_centered_loss(jnp.array([0, 1, 2, 3], dtype=jnp.int32), 0.1)

    def test_validate_losses_rejects_integer(self):
        with pytest.raises(TypeError):
            validate_losses(jnp.array([0, 1, 2, 3], dtype=jnp.int32))

    def test_float_losses_still_accepted(self):
        out = shape_centered_loss(jnp.array([0.0, 1.0, 2.0, 3.0]), 0.1)
        assert out.dtype == jnp.float32


class TestIssue15NonDictMapping:
    def test_mapping_proxy_type_tree_steps(self):
        params = {"block": MappingProxyType({"weight": jnp.ones((4, 4))})}
        manifest = Manifest(
            version=1,
            entries=(ManifestEntry(("block", "weight"), ParameterLayout.MATRIX, "w"),),
        )
        opt = ZeroGrad(
            optax.sgd(0.1),
            population_size=8,
            rank=2,
            sigma=0.1,
            seed=42,
            run_id="t",
            manifest=manifest,
        )
        state = opt.init(params)
        losses = jnp.arange(8, dtype=jnp.float32)
        new_params, new_state, _ = opt.step_from_losses(state, params, losses)
        assert new_state.generation == 1
        assert bool(jnp.all(jnp.isfinite(new_params["block"]["weight"])))


class TestIssue16NegativeArrayCandidate:
    def test_vmap_path_still_works(self):
        opt = _make_opt()
        model = Tiny(nnx.Rngs(0))
        opt.init(model)
        losses = opt.evaluate_shard(
            model, 0, _trivial_loss_fn, None, jnp.arange(8, dtype=jnp.int32)
        )
        assert losses.shape == (8,)


class TestIssue17FactorDtypeParity:
    def test_bf16_draw_matches_float32_draw_within_precision(self):
        key = jax.random.key(7)
        a16 = matrix_factors(key, (4, 2), rank=2, dtype=jnp.bfloat16)[0]
        a32 = matrix_factors(key, (4, 2), rank=2, dtype=jnp.float32)[0]
        np.testing.assert_allclose(
            np.asarray(a16, dtype=np.float32), np.asarray(a32), rtol=0.05
        )

    def test_bf16_draw_differs_from_direct_bf16_draw(self):
        key = jax.random.key(7)
        a32 = matrix_factors(key, (4, 2), rank=2, dtype=jnp.float32)[0]
        direct_bf16 = jax.random.normal(key, (4, 2), dtype=jnp.bfloat16)
        assert not np.allclose(
            np.asarray(direct_bf16, dtype=np.float32), np.asarray(a32), rtol=0.05
        )


class TestIssue21BoundedLossHistory:
    def test_default_is_finite(self):
        fc = FaultTolerantCluster(_make_opt(), _build_model, _loss_fn, seed=42, initial_nodes=1)
        assert fc.max_loss_history == DEFAULT_MAX_LOSS_HISTORY
        assert fc.max_loss_history > 0

    def test_explicit_zero_means_unlimited(self):
        fc = FaultTolerantCluster(
            _make_opt(), _build_model, _loss_fn, seed=42, initial_nodes=1, max_loss_history=0
        )
        for _ in range(5):
            fc.step(_batch())
        assert fc.loss_history_size == 5


class TestIssue23NodeStatusSlots:
    def test_slots_reject_undeclared_attribute(self):
        ns = NodeStatus(name="x", node=None)
        with pytest.raises(AttributeError):
            ns.undeclared = 1

    def test_shard_ids_field_present_and_default_none(self):
        ns = NodeStatus(name="x", node=None)
        assert ns._shard_ids is None

    def test_shard_ids_assignable_after_construction(self):
        ns = NodeStatus(name="x", node=None)
        ns._shard_ids = jnp.arange(4, dtype=jnp.int32)
        assert ns._shard_ids.shape == (4,)


class TestIssue24PartitionValidation:
    def test_compute_partition_sizes_rejects_zero_population(self):
        with pytest.raises(ValueError):
            compute_partition_sizes(0, [1.0, 1.0])

    def test_compute_partition_sizes_rejects_negative_population(self):
        with pytest.raises(ValueError):
            compute_partition_sizes(-4, [1.0, 1.0])

    def test_compute_partition_sizes_rejects_bool_population(self):
        with pytest.raises(ValueError):
            compute_partition_sizes(True, [1.0])

    def test_cluster_rejects_bool_num_nodes(self):
        with pytest.raises(ValueError):
            ClusterZeroGrad(_make_opt(), _build_model, _loss_fn, seed=42, num_nodes=True)

    def test_fault_tolerant_rejects_bool_initial_nodes(self):
        with pytest.raises(ValueError):
            FaultTolerantCluster(_make_opt(), _build_model, _loss_fn, seed=42, initial_nodes=True)


class TestIssue26PerCandidateRng:
    def test_candidates_receive_distinct_rng(self):
        opt = _make_opt()
        model = Tiny(nnx.Rngs(0))
        opt.init(model)

        def rng_loss(model, batch, rng):
            return jax.random.normal(rng, ()), None

        losses = opt.evaluate_shard(
            model, 0, rng_loss, None, jnp.arange(8, dtype=jnp.int32)
        )
        assert not bool(jnp.all(losses == losses[0]))
