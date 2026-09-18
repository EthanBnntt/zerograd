"""ZeroGrad optimizer surface: validation, replay parity, evaluate_shard, lifecycle."""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from _tiny import batch as _batch
from _tiny import make_model, make_opt
from _tiny import mse_loss as _loss_fn
from flax import nnx

from zerograd import (
    Manifest,
    ManifestEntry,
    ParameterLayout,
    ZeroGrad,
    candidate_key,
    group_key,
    shape_centered_loss,
    step_key,
)
from zerograd._factors import matrix_factors, scaled_factor, table_factors, vector_noise
from zerograd._nnx import params_pure_dict
from zerograd._optimizer import _build_pseudo_grad
from zerograd._replay import replay, replay_entry


def _manifest():
    return Manifest(version=1, entries=(
        ManifestEntry(("w",), ParameterLayout.MATRIX, "w"),
        ManifestEntry(("v",), ParameterLayout.VECTOR, "v"),
    ))


def _params():
    return {"w": jnp.ones((4, 2)), "v": jnp.zeros((2,))}


def _make_params():
    return {
        "linear": {"weight": jax.random.normal(jax.random.key(0), (8, 4))},
        "table": {"embed": jax.random.normal(jax.random.key(1), (16, 4))},
        "vector": {"scale": jax.random.normal(jax.random.key(2), (4,))},
    }


def _make_manifest():
    return Manifest(
        version=1,
        entries=(
            ManifestEntry(("linear", "weight"), ParameterLayout.MATRIX, "linear"),
            ManifestEntry(("table", "embed"), ParameterLayout.TABLE, "embed"),
            ManifestEntry(("vector", "scale"), ParameterLayout.VECTOR, "scale"),
        ),
    )


class Tiny(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.l = nnx.Linear(4, 2, rngs=rngs)

    def __call__(self, x=None):
        return self.l.kernel[...]


def _model():
    return Tiny(nnx.Rngs(0))


def _make_opt(**overrides):
    kwargs = {
        "transform": optax.adamw(0.01),
        "population_size": 4,
        "rank": 2,
        "sigma": 0.1,
        "seed": 42,
        "run_id": "test",
    }
    kwargs.update(overrides)
    return ZeroGrad(**kwargs)


def _trivial_loss_fn(model, batch):
    return jnp.sum(model.l.kernel[...] ** 2), None


class TestConstructorValidation:
    def test_rejects_non_transform_first_arg(self):
        bad_transform: Any = "not-a-transform"
        with pytest.raises(TypeError):
            ZeroGrad(bad_transform, 4, 2, 0.1, 42, "test")

    def test_rejects_population_size_below_two(self):
        with pytest.raises(ValueError):
            _make_opt(population_size=1)

    def test_rejects_bool_population_size(self):
        bad_population_size: Any = True
        with pytest.raises(ValueError):
            _make_opt(population_size=bad_population_size)

    def test_rejects_nonpositive_rank(self):
        with pytest.raises(ValueError):
            _make_opt(rank=0)

    def test_rejects_nonpositive_sigma(self):
        with pytest.raises(ValueError):
            _make_opt(sigma=0.0)

    def test_rejects_empty_run_id(self):
        with pytest.raises(ValueError):
            _make_opt(run_id="")

    def test_rejects_non_int_seed(self):
        bad_seed: Any = 1.5
        with pytest.raises(TypeError):
            _make_opt(seed=bad_seed)

    def test_config_properties(self):
        opt = _make_opt()
        assert opt.population_size == 4
        assert opt.rank == 2
        assert opt.sigma == 0.1
        assert opt.seed == 42
        assert opt.run_id == "test"
        assert _make_opt(manifest=_manifest()).manifest.version == 1


class TestInitAndStepValidation:
    def test_init_rejects_invalid_params(self):
        opt = _make_opt(manifest=_manifest())
        with pytest.raises(ValueError):
            opt.init({"w": jnp.ones((4,)), "v": jnp.zeros((2,))})

    def test_step_rejects_non_state(self):
        opt = _make_opt()
        bad_state: Any = "not-a-state"
        with pytest.raises(TypeError):
            opt.step(bad_state, _model(), None, _trivial_loss_fn)

    def test_step_rejects_non_module(self):
        opt = _make_opt(manifest=_manifest())
        state = opt.init(_params())
        bad_params: Any = _params()
        with pytest.raises(TypeError):
            opt.step(state, bad_params, None, _trivial_loss_fn)

    def test_step_from_losses_rejects_non_state(self):
        opt = _make_opt(manifest=_manifest())
        bad_state: Any = "not-a-state"
        with pytest.raises(TypeError):
            opt.step_from_losses(bad_state, _params(), jnp.zeros((4,)))

    def test_step_from_losses_rejects_wrong_loss_count(self):
        opt = _make_opt(manifest=_manifest())
        state = opt.init(_params())
        with pytest.raises(ValueError):
            opt.step_from_losses(state, _params(), jnp.zeros((3,)))

    def test_step_from_losses_rejects_non_finite_losses(self):
        opt = _make_opt(manifest=_manifest())
        state = opt.init(_params())
        with pytest.raises(ValueError):
            opt.step_from_losses(state, _params(), jnp.array([1.0, jnp.inf, 2.0, 3.0]))

    def test_step_from_losses_rejects_invalid_params(self):
        opt = _make_opt(manifest=_manifest())
        state = opt.init(_params())
        with pytest.raises(ValueError):
            opt.step_from_losses(state, {"w": jnp.ones((4,)), "v": jnp.zeros((2,))}, jnp.zeros((4,)))

    def test_step_rejects_non_finite_losses(self):
        opt = _make_opt()
        model = _model()
        state = opt.init(model)

        def inf_loss(model, batch):
            return jnp.asarray(jnp.inf), None

        with pytest.raises(ValueError):
            opt.step(state, model, None, inf_loss)


class TestEvaluateShard:
    def test_evaluate_shard_returns_one_loss_per_candidate(self):
        opt = _make_opt()
        model = _model()
        opt.init(model)
        ids = jnp.arange(4, dtype=jnp.int32)
        losses = opt.evaluate_shard(model, 0, _trivial_loss_fn, None, ids)
        assert losses.shape == (4,)

    def test_evaluate_shard_rejects_invalid_params(self):
        opt = _make_opt()
        bad_model: Any = "not-a-module"
        with pytest.raises(TypeError):
            opt.evaluate_shard(bad_model, 0, _trivial_loss_fn, None, jnp.arange(4, dtype=jnp.int32))

    def test_evaluate_shard_explicit_rng_used(self):
        opt = _make_opt()
        model = _model()
        opt.init(model)
        ids = jnp.arange(4, dtype=jnp.int32)
        losses = opt.evaluate_shard(model, 0, _trivial_loss_fn, None, ids, rng=jax.random.key(99))
        assert losses.shape == (4,)

    def test_evaluate_shard_matches_full_step(self):
        opt = _make_opt()
        model = _model()
        state = opt.init(model)
        ids = jnp.arange(4, dtype=jnp.int32)
        losses = opt.evaluate_shard(model, 0, _trivial_loss_fn, None, ids)
        p1, s1, m1 = opt.step_from_losses(state, model, losses)
        p2, s2, m2 = opt.step(state, model, None, _trivial_loss_fn)
        assert s1.generation == s2.generation == 1
        assert m1.population_size == m2.population_size
        for a, b in zip(
            jax.tree_util.tree_leaves(params_pure_dict(p1)),
            jax.tree_util.tree_leaves(params_pure_dict(p2)), strict=False,
        ):
            assert jnp.array_equal(a, b)

    def test_subset_losses_match_full_evaluation(self):
        opt = make_opt(pop=8, run_id="cov")
        model = make_model(0)
        opt.init(model)
        full = opt.evaluate_shard(model, 0, _loss_fn, _batch(), jnp.arange(8, dtype=jnp.int32))
        subset_ids = jnp.array([2, 5, 7], dtype=jnp.int32)
        sub = opt.evaluate_shard(model, 0, _loss_fn, _batch(), subset_ids)
        np.testing.assert_allclose(np.asarray(sub), np.asarray(full)[[2, 5, 7]], rtol=1e-6)

    def test_nnx_full_vmap_eval_finite(self):
        class VmapTiny(nnx.Module):
            def __init__(self, rngs: nnx.Rngs):
                self.l = nnx.Linear(4, 3, rngs=rngs)

            def __call__(self, x):
                return self.l(x)

        model = VmapTiny(nnx.Rngs(0))
        opt = ZeroGrad(
            optax.sgd(0.01), population_size=8, rank=1, sigma=0.05, seed=0, run_id="full-vmap",
        )
        opt.init(model)
        x = jax.random.normal(jax.random.key(1), (16, 4))
        y = jax.random.randint(jax.random.key(2), (16,), 0, 3)

        def loss_fn(m, b):
            bx, by = b
            return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(m(bx), by)), None

        losses = opt.evaluate_shard(
            model, 0, loss_fn, (x, y), jnp.arange(8, dtype=jnp.int32), rng=jax.random.key(3)
        )
        assert losses.shape == (8,)
        assert jnp.all(jnp.isfinite(losses))


class TestReplayParity:
    def test_matrix_replay_matches_dense(self):
        params = _make_params()
        manifest = _make_manifest()
        seed, run_id, gen = 42, "test", 0
        pop, rank, sigma = 8, 2, 0.1
        base_key = step_key(seed, run_id, gen, manifest.version)
        ids = jnp.arange(pop, dtype=jnp.int32)
        losses = jnp.array([float(i) for i in range(pop)])
        shaped = shape_centered_loss(losses, sigma)
        result = replay_entry(params, manifest, ("linear", "weight"), base_key, ids, shaped, rank)
        total = jnp.zeros((8, 4))
        for i in range(pop):
            ck = candidate_key(base_key, i)
            gk = group_key(ck, manifest, "linear")
            a, b = matrix_factors(gk, (8, 4), rank, dtype=jnp.float32)
            total = total + shaped[i] * (a @ b)
        expected = total * scaled_factor(rank, 1.0, jnp.float32)
        np.testing.assert_allclose(np.asarray(result), np.asarray(expected), rtol=1e-5, atol=1e-5)

    def test_table_replay_matches_dense(self):
        params = _make_params()
        manifest = _make_manifest()
        seed, run_id, gen = 42, "test", 0
        pop, rank, sigma = 8, 2, 0.1
        base_key = step_key(seed, run_id, gen, manifest.version)
        ids = jnp.arange(pop, dtype=jnp.int32)
        losses = jnp.array([float(i) for i in range(pop)])
        shaped = shape_centered_loss(losses, sigma)
        result = replay_entry(params, manifest, ("table", "embed"), base_key, ids, shaped, rank)
        total = jnp.zeros((16, 4))
        for i in range(pop):
            ck = candidate_key(base_key, i)
            gk = group_key(ck, manifest, "embed")
            a, b = table_factors(gk, (16, 4), rank, dtype=jnp.float32)
            total = total + shaped[i] * (a @ b.T)
        expected = total * scaled_factor(rank, 1.0, jnp.float32)
        np.testing.assert_allclose(np.asarray(result), np.asarray(expected), rtol=1e-5, atol=1e-5)

    def test_vector_replay_matches_dense(self):
        params = _make_params()
        manifest = _make_manifest()
        seed, run_id, gen = 42, "test", 0
        pop, sigma = 8, 0.1
        base_key = step_key(seed, run_id, gen, manifest.version)
        ids = jnp.arange(pop, dtype=jnp.int32)
        losses = jnp.array([float(i) for i in range(pop)])
        shaped = shape_centered_loss(losses, sigma)
        result = replay_entry(params, manifest, ("vector", "scale"), base_key, ids, shaped, 1)
        total = jnp.zeros((4,))
        for i in range(pop):
            ck = candidate_key(base_key, i)
            gk = group_key(ck, manifest, "scale")
            noise = vector_noise(gk, (4,), dtype=jnp.float32)
            total = total + shaped[i] * noise
        expected = total * scaled_factor(1, 1.0, jnp.float32)
        np.testing.assert_allclose(np.asarray(result), np.asarray(expected), rtol=1e-5, atol=1e-5)

    def test_replay_returns_nested_mapping(self):
        params = _make_params()
        manifest = _make_manifest()
        base_key = step_key(42, "test", 0, manifest.version)
        ids = jnp.arange(4, dtype=jnp.int32)
        shaped = shape_centered_loss(jnp.array([1.0, 2.0, 3.0, 4.0]), 0.1)
        result = replay(params, manifest, base_key, ids, shaped, 2)
        assert "linear" in result and "weight" in result["linear"]
        assert "table" in result and "embed" in result["table"]
        assert "vector" in result and "scale" in result["vector"]


class TestOptimizerLifecycle:
    def test_state_generation_advances(self):
        opt = _make_opt()
        model = _model()
        state = opt.init(model)
        assert state.generation == 0
        _, state, _ = opt.step(state, model, None, _trivial_loss_fn)
        assert state.generation == 1

    def test_step_descends_loss(self):
        class DescentTiny(nnx.Module):
            def __init__(self, rngs: nnx.Rngs):
                self.l = nnx.Linear(4, 4, use_bias=False, rngs=rngs)

            def __call__(self, x):
                return self.l(x)

        model = DescentTiny(nnx.Rngs(0))
        model.l.kernel[...] = jnp.ones((4, 4))
        opt = ZeroGrad(
            optax.sgd(learning_rate=0.1),
            population_size=64, rank=4, sigma=0.05, seed=42, run_id="test",
        )
        state = opt.init(model)
        x = jnp.ones((1, 4))

        def loss_fn(model, batch):
            return jnp.sum(model(batch) ** 2), None

        loss_before = float(jnp.sum(model(x) ** 2))
        new_model, _, _ = opt.step(state, model, x, loss_fn)
        loss_after = float(jnp.sum(new_model(x) ** 2))
        assert loss_after < loss_before

    def test_non_manifest_params_get_zero_gradient(self):
        params = {
            "linear": {"weight": jnp.ones((8, 4))},
            "frozen": {"weight": jnp.ones((8, 4))},
        }
        manifest = Manifest(
            version=1,
            entries=(ManifestEntry(("linear", "weight"), ParameterLayout.MATRIX, "linear"),),
        )
        opt = ZeroGrad(
            optax.adamw(1.0, weight_decay=0.0),
            population_size=4, rank=2, sigma=0.1, seed=42, run_id="test", manifest=manifest,
        )
        state = opt.init(params)
        losses = jnp.arange(4, dtype=jnp.float32)
        new_params, _, _ = opt.step_from_losses(state, params, losses)
        np.testing.assert_allclose(
            np.asarray(new_params["frozen"]["weight"]),
            np.asarray(params["frozen"]["weight"]),
            rtol=1e-6, atol=1e-6,
        )

    def test_all_equal_losses_no_op_for_sgd(self):
        opt = ZeroGrad(
            optax.sgd(0.1),
            population_size=8, rank=2, sigma=0.1, seed=42, run_id="noop",
        )
        model = make_model(0)
        state = opt.init(model)
        before = params_pure_dict(model)
        losses = jnp.full((8,), 1.5)
        new_model, _, _ = opt.step_from_losses(state, model, losses)
        after = params_pure_dict(new_model)
        np.testing.assert_array_equal(
            np.asarray(after["l"]["kernel"]), np.asarray(before["l"]["kernel"])
        )


class TestCrossInstanceDeterminism:
    def test_two_instances_produce_identical_trajectory(self):
        def make():
            return ZeroGrad(
                optax.adamw(0.01),
                population_size=8, rank=2, sigma=0.1, seed=42, run_id="det",
            )

        opt1, opt2 = make(), make()
        m1, m2 = make_model(0), make_model(0)
        s1, s2 = opt1.init(m1), opt2.init(m2)
        batch = _batch()
        for _ in range(3):
            m1, s1, _ = opt1.step(s1, m1, batch, _loss_fn)
            m2, s2, _ = opt2.step(s2, m2, batch, _loss_fn)
        np.testing.assert_array_equal(
            np.asarray(params_pure_dict(m1)["l"]["kernel"]),
            np.asarray(params_pure_dict(m2)["l"]["kernel"]),
        )


class TestPseudoGradNegation:
    def test_passthrough_for_non_array_non_dict_values(self):
        params = {"w": jnp.ones((2, 2)), "meta": "unchanged"}
        descent = {"w": jnp.ones((2, 2))}
        result = _build_pseudo_grad(descent, params)
        assert result["meta"] == "unchanged"
        assert float(jnp.sum(result["w"])) == -4.0

    def test_non_manifest_array_gets_zeros(self):
        params = {"w": jnp.ones((2, 2)), "frozen": jnp.ones((2, 2))}
        descent = {"w": jnp.ones((2, 2))}
        result = _build_pseudo_grad(descent, params)
        assert float(jnp.sum(result["frozen"])) == 0.0

    def test_nested_dict_descended(self):
        params = {"layer": {"w": jnp.ones((2, 2)), "extra": jnp.ones((2, 2))}}
        descent = {"layer": {"w": jnp.ones((2, 2))}}
        result = _build_pseudo_grad(descent, params)
        assert float(jnp.sum(result["layer"]["w"])) == -4.0
        assert float(jnp.sum(result["layer"]["extra"])) == 0.0


class TestStateAndMetrics:
    def test_step_metrics_report_population_diagnostics(self):
        opt = _make_opt()
        model = _model()
        state = opt.init(model)
        _, _, metrics = opt.step(state, model, None, _trivial_loss_fn)
        assert metrics.generation == 0
        assert metrics.population_size == 4
        assert metrics.min_loss <= metrics.mean_loss <= metrics.max_loss

    def test_metrics_match_loss_array(self):
        opt = make_opt(pop=8, run_id="cov")
        model = make_model(0)
        state = opt.init(model)
        losses = opt.evaluate_shard(model, 0, _loss_fn, _batch(), jnp.arange(8, dtype=jnp.int32))
        _, _, m = opt.step_from_losses(state, model, losses)
        assert m.mean_loss == pytest.approx(float(jnp.mean(losses)))
        assert m.min_loss == pytest.approx(float(jnp.min(losses)))
        assert m.max_loss == pytest.approx(float(jnp.max(losses)))
        assert m.population_size == 8
