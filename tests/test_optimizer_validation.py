"""Validation and lifecycle tests for the ZeroGrad optimizer surface."""

import jax
import jax.numpy as jnp
import optax
import pytest

from zerograd import Manifest, ManifestEntry, ParameterLayout, ZeroGrad, ZeroGradState
from zerograd._optimizer import _build_pseudo_grad


def _manifest():
    return Manifest(version=1, entries=(
        ManifestEntry(("w",), ParameterLayout.MATRIX, "w"),
        ManifestEntry(("v",), ParameterLayout.VECTOR, "v"),
    ))


def _params():
    return {"w": jnp.ones((4, 2)), "v": jnp.zeros((2,))}


def _make_opt(**overrides):
    kwargs = dict(
        manifest=_manifest(), transform=optax.adamw(0.01),
        population_size=4, rank=2, sigma=0.1, seed=42, run_id="test",
    )
    kwargs.update(overrides)
    return ZeroGrad(**kwargs)


def _trivial_loss_fn(p, candidate, batch, rng):
    return jnp.sum(p["w"] ** 2) + jnp.sum(p["v"] ** 2), None


class TestConstructorValidation:
    def test_rejects_non_manifest(self):
        with pytest.raises(TypeError):
            ZeroGrad("not-a-manifest", optax.adamw(0.01), 4, 2, 0.1, 42, "test")  # type: ignore[arg-type]

    def test_rejects_population_size_below_two(self):
        with pytest.raises(ValueError):
            _make_opt(population_size=1)

    def test_rejects_bool_population_size(self):
        with pytest.raises(ValueError):
            _make_opt(population_size=True)  # type: ignore[arg-type]

    def test_rejects_nonpositive_rank(self):
        with pytest.raises(ValueError):
            _make_opt(rank=0)

    def test_rejects_nonpositive_sigma(self):
        with pytest.raises(ValueError):
            _make_opt(sigma=0.0)

    def test_rejects_non_float_sigma(self):
        with pytest.raises(ValueError):
            _make_opt(sigma=1)  # type: ignore[arg-type]

    def test_rejects_bool_seed(self):
        with pytest.raises(TypeError):
            _make_opt(seed=True)  # type: ignore[arg-type]

    def test_rejects_non_integer_seed(self):
        with pytest.raises(TypeError):
            _make_opt(seed=1.5)  # type: ignore[arg-type]

    def test_rejects_empty_run_id(self):
        with pytest.raises(ValueError):
            _make_opt(run_id="")

    def test_rejects_non_string_run_id(self):
        with pytest.raises(ValueError):
            _make_opt(run_id=None)  # type: ignore[arg-type]


class TestConfigurationProperties:
    def test_properties_expose_constructor_values(self):
        opt = _make_opt()
        assert opt.population_size == 4
        assert opt.rank == 2
        assert opt.sigma == 0.1
        assert opt.seed == 42
        assert opt.run_id == "test"
        assert opt.manifest is _manifest() or opt.manifest.version == 1


class TestInitAndStepValidation:
    def test_init_rejects_invalid_params(self):
        opt = _make_opt()
        with pytest.raises(ValueError):
            opt.init({"w": jnp.ones((4,)), "v": jnp.zeros((2,))})

    def test_step_rejects_non_state(self):
        opt = _make_opt()
        with pytest.raises(TypeError):
            opt.step("not-a-state", _params(), None, _trivial_loss_fn)  # type: ignore[arg-type]

    def test_step_from_losses_rejects_non_state(self):
        opt = _make_opt()
        with pytest.raises(TypeError):
            opt.step_from_losses("not-a-state", _params(), jnp.zeros((4,)))  # type: ignore[arg-type]

    def test_step_from_losses_rejects_wrong_loss_count(self):
        opt = _make_opt()
        state = opt.init(_params())
        with pytest.raises(ValueError):
            opt.step_from_losses(state, _params(), jnp.zeros((3,)))

    def test_step_from_losses_rejects_non_finite_losses(self):
        opt = _make_opt()
        state = opt.init(_params())
        with pytest.raises(ValueError):
            opt.step_from_losses(state, _params(), jnp.array([1.0, jnp.inf, 2.0, 3.0]))

    def test_step_from_losses_rejects_invalid_params(self):
        opt = _make_opt()
        state = opt.init(_params())
        with pytest.raises(ValueError):
            opt.step_from_losses(state, {"w": jnp.ones((4,)), "v": jnp.zeros((2,))}, jnp.zeros((4,)))


class TestEvaluateShard:
    def test_evaluate_shard_returns_one_loss_per_candidate(self):
        opt = _make_opt()
        params = _params()
        ids = jnp.arange(4, dtype=jnp.int32)
        losses = opt.evaluate_shard(params, 0, _trivial_loss_fn, None, ids)
        assert losses.shape == (4,)

    def test_evaluate_shard_rejects_invalid_params(self):
        opt = _make_opt()
        with pytest.raises(ValueError):
            opt.evaluate_shard({"w": jnp.ones((4,))}, 0, _trivial_loss_fn, None, jnp.arange(4, dtype=jnp.int32))

    def test_evaluate_shard_explicit_rng_used(self):
        opt = _make_opt()
        params = _params()
        ids = jnp.arange(4, dtype=jnp.int32)
        losses = opt.evaluate_shard(params, 0, _trivial_loss_fn, None, ids, rng=jax.random.key(99))
        assert losses.shape == (4,)

    def test_evaluate_shard_matches_full_step(self):
        """step() == evaluate_shard(all ids) then step_from_losses."""
        opt = _make_opt()
        params = _params()
        state = opt.init(params)
        ids = jnp.arange(4, dtype=jnp.int32)
        losses = opt.evaluate_shard(params, 0, _trivial_loss_fn, None, ids)
        p1, s1, m1 = opt.step_from_losses(state, params, losses)
        p2, s2, m2 = opt.step(state, params, None, _trivial_loss_fn)
        assert s1.generation == s2.generation == 1
        assert m1.population_size == m2.population_size
        for a, b in zip(jax.tree_util.tree_leaves(p1), jax.tree_util.tree_leaves(p2)):
            assert jnp.array_equal(a, b)


class TestPseudoGradNegation:
    def test_passthrough_for_non_array_non_dict_values(self):
        # _build_pseudo_grad walks params; non-array, non-dict leaves pass through.
        params = {"w": jnp.ones((2, 2)), "meta": "unchanged"}
        descent = {"w": jnp.ones((2, 2))}
        result = _build_pseudo_grad(descent, params)
        # manifest leaf negated, non-array leaf passed through unchanged.
        assert result["meta"] == "unchanged"
        assert float(jnp.sum(result["w"])) == -4.0

    def test_non_manifest_array_gets_zeros(self):
        params = {"w": jnp.ones((2, 2)), "frozen": jnp.ones((2, 2))}
        descent = {"w": jnp.ones((2, 2))}  # frozen is not in descent
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
        params = _params()
        state = opt.init(params)
        _, _, metrics = opt.step(state, params, None, _trivial_loss_fn)
        assert metrics.generation == 0
        assert metrics.population_size == 4
        assert metrics.min_loss <= metrics.mean_loss <= metrics.max_loss

    def test_state_generation_advances(self):
        opt = _make_opt()
        params = _params()
        state = opt.init(params)
        assert state.generation == 0
        _, state, _ = opt.step(state, params, None, _trivial_loss_fn)
        assert state.generation == 1
