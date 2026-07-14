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

from zerograd import (
    DistributedZeroGrad,
    Manifest,
    ManifestEntry,
    ParameterLayout,
    ZeroGrad,
    matrix_factors,
    table_factors,
    vector_noise,
)


# ── Shared helpers ───────────────────────────────────────────────────────────

def _manifest():
    return Manifest(version=1, entries=(
        ManifestEntry(("w",), ParameterLayout.MATRIX, "w"),
    ))


def _params():
    return {"w": jnp.ones((4, 2))}


def _loss_fn(p, candidate, batch, rng):
    return jnp.sum(candidate.linear(p, ("w",), batch) ** 2), None


def _make_opt(pop=8, **kw):
    defaults = dict(
        manifest=_manifest(), transform=optax.adamw(0.01),
        population_size=pop, rank=2, sigma=0.1, seed=42, run_id="cov",
    )
    defaults.update(kw)
    return ZeroGrad(**defaults)


def _batch():
    return jnp.ones((3, 4))


def _a_device():
    return jax.devices()[0]


def _key(seed):
    return jax.random.key(seed)


# ── 1. Distributed gather ordering ──────────────────────────────────────────

class TestDistributedGatherOrdering:
    def test_multi_device_matches_single_device_exactly(self):
        opt = _make_opt(pop=8)
        params = _params()
        state = opt.init(params)
        batch = _batch()
        ref_params, _, _ = opt.step(state, params, batch, _loss_fn)

        coord = DistributedZeroGrad(opt, [_a_device(), _a_device()], _loss_fn, weights=[1.0, 3.0])
        dstate = coord.init(params)
        got_params, new_state, _ = coord.step(dstate, params, batch)
        assert new_state.generation == 1
        np.testing.assert_allclose(np.asarray(got_params["w"]), np.asarray(ref_params["w"]), rtol=1e-5)

    def test_uneven_split_matches_single_device(self):
        # pop=7 is not divisible by 2 devices; remainder must still land in order.
        opt = _make_opt(pop=7)
        params = _params()
        state = opt.init(params)
        batch = _batch()
        ref_params, _, _ = opt.step(state, params, batch, _loss_fn)

        coord = DistributedZeroGrad(opt, [_a_device(), _a_device()], _loss_fn, weights=[1.0, 2.0])
        dstate = coord.init(params)
        got_params, _, _ = coord.step(dstate, params, batch)
        np.testing.assert_allclose(np.asarray(got_params["w"]), np.asarray(ref_params["w"]), rtol=1e-5)


# ── 2. Factor dtype is honoured ─────────────────────────────────────────────

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


# ── 3. Multi-step cross-instance determinism ─────────────────────────────────

class TestCrossInstanceDeterminism:
    def test_two_instances_produce_identical_trajectory(self):
        manifest = _manifest()

        def make():
            return ZeroGrad(
                manifest, optax.adamw(0.01),
                population_size=8, rank=2, sigma=0.1, seed=42, run_id="det",
            )

        opt1, opt2 = make(), make()
        params1 = params2 = _params()
        s1, s2 = opt1.init(params1), opt2.init(params2)
        batch = _batch()
        for _ in range(3):
            params1, s1, _ = opt1.step(s1, params1, batch, _loss_fn)
            params2, s2, _ = opt2.step(s2, params2, batch, _loss_fn)
        np.testing.assert_array_equal(np.asarray(params1["w"]), np.asarray(params2["w"]))


# ── 4. evaluate_shard subset ────────────────────────────────────────────────

class TestEvaluateShardSubset:
    def test_subset_losses_match_full_evaluation(self):
        opt = _make_opt(pop=8)
        params = _params()
        full = opt.evaluate_shard(params, 0, _loss_fn, _batch(), jnp.arange(8, dtype=jnp.int32))
        subset_ids = jnp.array([2, 5, 7], dtype=jnp.int32)
        sub = opt.evaluate_shard(params, 0, _loss_fn, _batch(), subset_ids)
        np.testing.assert_allclose(np.asarray(sub), np.asarray(full)[[2, 5, 7]], rtol=1e-6)


# ── 5. StepMetrics reflect actual losses ────────────────────────────────────

class TestStepMetricsAgainstLosses:
    def test_metrics_match_loss_array(self):
        opt = _make_opt(pop=8)
        params = _params()
        state = opt.init(params)
        losses = opt.evaluate_shard(params, 0, _loss_fn, _batch(), jnp.arange(8, dtype=jnp.int32))
        _, _, m = opt.step_from_losses(state, params, losses)
        assert m.mean_loss == pytest.approx(float(jnp.mean(losses)))
        assert m.min_loss == pytest.approx(float(jnp.min(losses)))
        assert m.max_loss == pytest.approx(float(jnp.max(losses)))
        assert m.population_size == 8


# ── 6. step() after calibrate() ─────────────────────────────────────────────

class TestStepAfterCalibrate:
    def test_step_is_consistent_after_calibration(self):
        opt = _make_opt(pop=8)
        coord = DistributedZeroGrad(opt, [_a_device(), _a_device()], _loss_fn)
        coord.calibrate(_params(), _batch(), warmup=0, trials=1)
        dstate = coord.init(_params())
        new_params, new_state, _ = coord.step(dstate, _params(), _batch())
        assert new_state.generation == 1
        assert bool(jnp.all(jnp.isfinite(new_params["w"])))
        assert sum(coord.partition_sizes) == 8


# ── 7. table_factors validation/determinism parity ──────────────────────────

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


# ── 8. All-equal losses no-op ───────────────────────────────────────────────

class TestAllEqualLossesNoOp:
    def test_sgd_leaves_params_unchanged_when_losses_equal(self):
        opt = ZeroGrad(
            _manifest(), optax.sgd(0.1),
            population_size=8, rank=2, sigma=0.1, seed=42, run_id="noop",
        )
        params = _params()
        state = opt.init(params)
        losses = jnp.full((8,), 1.5)  # all-equal → centered weights are zero
        new_params, _, _ = opt.step_from_losses(state, params, losses)
        np.testing.assert_array_equal(np.asarray(new_params["w"]), np.asarray(params["w"]))
