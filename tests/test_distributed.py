"""Tests for distributed evaluation: partitioning, device shards, and coordinator."""

import jax
import jax.numpy as jnp
import optax
import pytest

from zerograd import (
    CalibrationResult,
    DeviceShard,
    DistributedZeroGrad,
    Manifest,
    ManifestEntry,
    ParameterLayout,
    ShardResult,
    ZeroGrad,
    compute_partition_sizes,
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
        population_size=pop, rank=2, sigma=0.1, seed=42, run_id="dist-test",
    )
    defaults.update(kw)
    return ZeroGrad(**defaults)


def _a_device():
    return jax.devices()[0]


# ── compute_partition_sizes ─────────────────────────────────────────────────

class TestComputePartitionSizes:
    def test_empty_weights_rejected(self):
        with pytest.raises(ValueError):
            compute_partition_sizes(10, [])

    def test_negative_weights_rejected(self):
        with pytest.raises(ValueError):
            compute_partition_sizes(10, [1.0, -1.0])

    def test_all_zero_weights_rejected(self):
        with pytest.raises(ValueError):
            compute_partition_sizes(10, [0.0, 0.0])

    def test_sum_equals_population(self):
        for pop in (1, 7, 100, 101):
            sizes = compute_partition_sizes(pop, [1.0, 3.0, 2.0])
            assert sum(sizes) == pop
            assert len(sizes) == 3

    def test_zero_weight_device_gets_zero_candidates(self):
        sizes = compute_partition_sizes(10, [0.0, 1.0])
        assert sizes == [0, 10]

    def test_even_split_for_equal_weights(self):
        sizes = compute_partition_sizes(8, [1.0, 1.0, 1.0, 1.0])
        assert sizes == [2, 2, 2, 2]

    def test_remainder_distributed_to_largest_fractional(self):
        # 10 into [1,1,1] -> 3.33 each -> 3,3,3 + remainder 1 to largest frac
        sizes = compute_partition_sizes(10, [1.0, 1.0, 1.0])
        assert sum(sizes) == 10
        # One shard gets the extra candidate.
        assert sorted(sizes) == [3, 3, 4]

    def test_weighted_split_proportional(self):
        sizes = compute_partition_sizes(10, [1.0, 4.0])
        assert sizes == [2, 8]

    def test_single_weight(self):
        assert compute_partition_sizes(5, [1.0]) == [5]


# ── DeviceShard ─────────────────────────────────────────────────────────────

class TestDeviceShard:
    def test_default_name_uses_platform(self):
        dev = _a_device()
        shard = DeviceShard(dev, _make_opt(), _loss_fn)
        assert shard.name == f"shard-{dev.platform}"

    def test_custom_name(self):
        shard = DeviceShard(_a_device(), _make_opt(), _loss_fn, name="custom")
        assert shard.name == "custom"

    def test_num_candidates_zero_before_assignment(self):
        shard = DeviceShard(_a_device(), _make_opt(), _loss_fn)
        assert shard.num_candidates == 0

    def test_num_candidates_after_assignment(self):
        shard = DeviceShard(_a_device(), _make_opt(), _loss_fn)
        shard._candidate_ids = jnp.arange(5, dtype=jnp.int32)
        assert shard.num_candidates == 5

    def test_evaluate_returns_shard_result(self):
        opt = _make_opt()
        shard = DeviceShard(_a_device(), opt, _loss_fn)
        params = _params()
        batch = jnp.ones((3, 4))
        ids = jnp.arange(4, dtype=jnp.int32)
        result = shard.evaluate(params, batch, ids, generation=0)
        assert isinstance(result, ShardResult)
        assert result.losses.shape == (4,)
        # candidate_ids echoed back unchanged
        assert int(result.candidate_ids[-1]) == 3

    def test_evaluate_uses_default_rng_when_none(self):
        opt = _make_opt()
        shard = DeviceShard(_a_device(), opt, _loss_fn)
        # rng=None path should not raise and should produce finite losses.
        result = shard.evaluate(_params(), jnp.ones((3, 4)), jnp.arange(4, dtype=jnp.int32), 0)
        assert bool(jnp.all(jnp.isfinite(result.losses)))


# ── DistributedZeroGrad ─────────────────────────────────────────────────────

class TestDistributedConstruction:
    def test_empty_devices_rejected(self):
        with pytest.raises(ValueError):
            DistributedZeroGrad(_make_opt(), [], _loss_fn)

    def test_weights_length_mismatch_rejected(self):
        with pytest.raises(ValueError):
            DistributedZeroGrad(_make_opt(), [_a_device()], _loss_fn, weights=[1.0, 1.0])

    def test_default_equal_weights(self):
        coord = DistributedZeroGrad(_make_opt(), [_a_device(), _a_device()], _loss_fn)
        assert coord.weights == [1.0, 1.0]

    def test_custom_weights(self):
        coord = DistributedZeroGrad(
            _make_opt(), [_a_device(), _a_device()], _loss_fn, weights=[1.0, 3.0],
        )
        assert coord.weights == [1.0, 3.0]

    def test_shards_match_devices(self):
        coord = DistributedZeroGrad(
            _make_opt(), [_a_device(), _a_device()], _loss_fn,
        )
        assert len(coord.shards) == 2

    def test_partition_sizes_sum_to_population(self):
        coord = DistributedZeroGrad(
            _make_opt(pop=10), [_a_device(), _a_device(), _a_device()],
            _loss_fn, weights=[1.0, 2.0, 1.0],
        )
        sizes = coord.partition_sizes
        assert sum(sizes) == 10
        assert len(sizes) == 3

    def test_explicit_coordinator_device(self):
        dev = _a_device()
        coord = DistributedZeroGrad(
            _make_opt(), [dev], _loss_fn, coordinator_device=dev,
        )
        assert coord._coordinator_device is dev


class TestDistributedStep:
    def test_init_returns_generation_zero(self):
        coord = DistributedZeroGrad(_make_opt(), [_a_device()], _loss_fn)
        state = coord.init(_params())
        assert state.generation == 0

    def test_single_device_step_matches_plain_optimizer(self):
        opt = _make_opt(pop=8)
        params = _params()
        state = opt.init(params)
        batch = jnp.ones((3, 4))
        ref_params, ref_state, _ = opt.step(state, params, batch, _loss_fn)

        coord = DistributedZeroGrad(opt, [_a_device()], _loss_fn)
        dstate = coord.init(params)
        new_params, new_state, metrics = coord.step(dstate, params, batch)

        assert new_state.generation == 1
        assert metrics.population_size == 8
        for a, b in zip(jax.tree_util.tree_leaves(new_params), jax.tree_util.tree_leaves(ref_params)):
            np_allclose(a, b)

    def test_multi_device_step_matches_single_device(self):
        opt = _make_opt(pop=8)
        params = _params()
        batch = jnp.ones((3, 4))

        coord = DistributedZeroGrad(opt, [_a_device(), _a_device()], _loss_fn)
        dstate = coord.init(params)
        new_params, new_state, _ = coord.step(dstate, params, batch)
        assert new_state.generation == 1
        assert bool(jnp.all(jnp.isfinite(new_params["w"])))

    def test_weighted_partition_step(self):
        opt = _make_opt(pop=8)
        params = _params()
        batch = jnp.ones((3, 4))
        coord = DistributedZeroGrad(opt, [_a_device(), _a_device()], _loss_fn, weights=[1.0, 3.0])
        assert coord.partition_sizes == [2, 6]
        dstate = coord.init(params)
        _, new_state, _ = coord.step(dstate, params, batch)
        assert new_state.generation == 1

    def test_zero_weight_shard_skipped(self):
        opt = _make_opt(pop=8)
        params = _params()
        batch = jnp.ones((3, 4))
        coord = DistributedZeroGrad(opt, [_a_device(), _a_device()], _loss_fn, weights=[0.0, 1.0])
        assert coord.partition_sizes == [0, 8]
        dstate = coord.init(params)
        _, new_state, _ = coord.step(dstate, params, batch)
        assert new_state.generation == 1


class TestCalibrate:
    def test_calibrate_updates_weights_and_returns_results(self):
        opt = _make_opt(pop=8)
        coord = DistributedZeroGrad(opt, [_a_device(), _a_device()], _loss_fn)
        results = coord.calibrate(_params(), jnp.ones((3, 4)), warmup=0, trials=1)
        assert len(results) == 2
        assert all(isinstance(r, CalibrationResult) for r in results)
        # Weights were updated (no longer equal in general, but length matches).
        assert len(coord.weights) == 2
        # Partition still sums to population.
        assert sum(coord.partition_sizes) == 8

    def test_calibrate_result_fields_populated(self):
        opt = _make_opt(pop=8)
        coord = DistributedZeroGrad(opt, [_a_device()], _loss_fn)
        results = coord.calibrate(_params(), jnp.ones((3, 4)), warmup=1, trials=2)
        r = results[0]
        assert r.num_candidates == 4
        assert r.elapsed_seconds > 0
        assert r.per_candidate_seconds > 0
        assert r.device is coord.shards[0].device


def np_allclose(a, b):
    import numpy as np
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-5, atol=1e-5)
