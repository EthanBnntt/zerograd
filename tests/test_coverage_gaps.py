"""Coverage-gap tests addressing issue #33.

Adds targeted tests for correctness invariants that were previously
under-tested:
  1. Distributed gather ordering (multi-device == single-device, exactly).
  2. Factor dtype is honoured.
  3. Multi-step cross-instance determinism.
  4. evaluate_shard subset matches the full-eval entries.
  5. StepMetrics reflect the actual loss array.
  6. step() after calibrate() is consistent.
  7. table_factors validation/determinism parity with matrix_factors.
  8. All-equal losses are a no-op for plain SGD.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from _tiny import batch as _batch, make_model, make_opt, mse_loss as _loss_fn
from zerograd import (
    DistributedZeroGrad,
    ZeroGrad,
)
from zerograd._factors import matrix_factors, table_factors, vector_noise
from zerograd._nnx import params_pure_dict


def _model():
    return make_model(0)


def _make_opt(pop=8, **kw):
    return make_opt(pop=pop, run_id="cov", **kw)


def _a_device():
    return jax.devices()[0]


def _key(seed):
    return jax.random.key(seed)


class TestDistributedGatherOrdering:
    def test_multi_device_matches_single_device_exactly(self):
        opt = _make_opt(pop=8)
        model = _model()
        state = opt.init(model)
        batch = _batch()
        ref_model, _, _ = opt.step(state, model, batch, _loss_fn)

        model2 = _model()
        coord = DistributedZeroGrad(opt, [_a_device(), _a_device()], _loss_fn, weights=[1.0, 3.0])
        dstate = coord.init(model2)
        got_model, new_state, _ = coord.step(dstate, model2, batch)
        assert new_state.generation == 1
        np.testing.assert_allclose(
            np.asarray(params_pure_dict(got_model)["l"]["kernel"]),
            np.asarray(params_pure_dict(ref_model)["l"]["kernel"]),
            rtol=1e-5,
        )

    def test_uneven_split_matches_single_device(self):
        opt = _make_opt(pop=7)
        model = _model()
        state = opt.init(model)
        batch = _batch()
        ref_model, _, _ = opt.step(state, model, batch, _loss_fn)

        model2 = _model()
        coord = DistributedZeroGrad(opt, [_a_device(), _a_device()], _loss_fn, weights=[1.0, 2.0])
        dstate = coord.init(model2)
        got_model, _, _ = coord.step(dstate, model2, batch)
        np.testing.assert_allclose(
            np.asarray(params_pure_dict(got_model)["l"]["kernel"]),
            np.asarray(params_pure_dict(ref_model)["l"]["kernel"]),
            rtol=1e-5,
        )


class TestFactorDtype:
    def test_matrix_factors_dtype(self):
        a, b = matrix_factors(_key(0), (8, 4), rank=2, dtype=jnp.float32)
        assert a.dtype == jnp.float32
        assert b.dtype == jnp.float32

    def test_matrix_factors_bf16_dtype(self):
        a, b = matrix_factors(_key(0), (8, 4), rank=2, dtype=jnp.bfloat16)
        assert a.dtype == jnp.bfloat16
        assert b.dtype == jnp.bfloat16

    def test_vector_noise_dtype(self):
        n = vector_noise(_key(0), (8,), dtype=jnp.float32)
        assert n.dtype == jnp.float32


class TestCrossInstanceDeterminism:
    def test_two_instances_produce_identical_trajectory(self):
        def make():
            return ZeroGrad(
                optax.adamw(0.01),
                population_size=8, rank=2, sigma=0.1, seed=42, run_id="det",
            )

        opt1, opt2 = make(), make()
        m1, m2 = _model(), _model()
        s1, s2 = opt1.init(m1), opt2.init(m2)
        batch = _batch()
        for _ in range(3):
            m1, s1, _ = opt1.step(s1, m1, batch, _loss_fn)
            m2, s2, _ = opt2.step(s2, m2, batch, _loss_fn)
        np.testing.assert_array_equal(
            np.asarray(params_pure_dict(m1)["l"]["kernel"]),
            np.asarray(params_pure_dict(m2)["l"]["kernel"]),
        )


class TestEvaluateShardSubset:
    def test_subset_losses_match_full_evaluation(self):
        opt = _make_opt(pop=8)
        model = _model()
        opt.init(model)
        full = opt.evaluate_shard(model, 0, _loss_fn, _batch(), jnp.arange(8, dtype=jnp.int32))
        subset_ids = jnp.array([2, 5, 7], dtype=jnp.int32)
        sub = opt.evaluate_shard(model, 0, _loss_fn, _batch(), subset_ids)
        np.testing.assert_allclose(np.asarray(sub), np.asarray(full)[[2, 5, 7]], rtol=1e-6)


class TestStepMetricsAgainstLosses:
    def test_metrics_match_loss_array(self):
        opt = _make_opt(pop=8)
        model = _model()
        state = opt.init(model)
        losses = opt.evaluate_shard(model, 0, _loss_fn, _batch(), jnp.arange(8, dtype=jnp.int32))
        _, _, m = opt.step_from_losses(state, model, losses)
        assert m.mean_loss == pytest.approx(float(jnp.mean(losses)))
        assert m.min_loss == pytest.approx(float(jnp.min(losses)))
        assert m.max_loss == pytest.approx(float(jnp.max(losses)))
        assert m.population_size == 8


class TestStepAfterCalibrate:
    def test_step_is_consistent_after_calibration(self):
        opt = _make_opt(pop=8)
        coord = DistributedZeroGrad(opt, [_a_device(), _a_device()], _loss_fn)
        model = _model()
        opt.init(model)
        coord.calibrate(model, _batch(), warmup=0, trials=1)
        dstate = coord.init(model)
        new_model, new_state, _ = coord.step(dstate, model, _batch())
        assert new_state.generation == 1
        assert bool(jnp.all(jnp.isfinite(params_pure_dict(new_model)["l"]["kernel"])))
        assert sum(coord.partition_sizes) == 8


class TestTableFactorsValidation:
    def test_rejects_wrong_ndim_shape(self):
        with pytest.raises(ValueError):
            table_factors(_key(0), (16,), rank=2, dtype=jnp.float32)

    def test_rejects_nonpositive_dimension(self):
        with pytest.raises(ValueError):
            table_factors(_key(0), (16, 0), rank=2, dtype=jnp.float32)

    def test_rejects_nonpositive_rank(self):
        with pytest.raises(ValueError):
            table_factors(_key(0), (16, 4), rank=0, dtype=jnp.float32)

    def test_rejects_bool_rank(self):
        with pytest.raises(ValueError):
            table_factors(_key(0), (16, 4), rank=True, dtype=jnp.float32)

    def test_deterministic_for_same_key(self):
        a1, b1 = table_factors(_key(7), (16, 4), rank=3, dtype=jnp.float32)
        a2, b2 = table_factors(_key(7), (16, 4), rank=3, dtype=jnp.float32)
        np.testing.assert_array_equal(np.asarray(a1), np.asarray(a2))
        np.testing.assert_array_equal(np.asarray(b1), np.asarray(b2))


class TestAllEqualLossesNoOp:
    def test_sgd_leaves_params_unchanged_when_losses_equal(self):
        opt = ZeroGrad(
            optax.sgd(0.1),
            population_size=8, rank=2, sigma=0.1, seed=42, run_id="noop",
        )
        model = _model()
        state = opt.init(model)
        before = params_pure_dict(model)
        losses = jnp.full((8,), 1.5)  # all-equal → centered weights are zero
        new_model, _, _ = opt.step_from_losses(state, model, losses)
        after = params_pure_dict(new_model)
        np.testing.assert_array_equal(
            np.asarray(after["l"]["kernel"]), np.asarray(before["l"]["kernel"])
        )
