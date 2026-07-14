"""Tests for factor replay parity and ZeroGrad optimizer lifecycle."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from zerograd import (
    Manifest,
    ManifestEntry,
    ParameterLayout,
    ZeroGrad,
    candidate_key,
    group_key,
    replay,
    replay_entry,
    shape_centered_loss,
    step_key,
)
from zerograd._factors import matrix_factors, scaled_factor, table_factors, vector_noise


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
    def test_init_returns_generation_zero(self):
        params = _make_params()
        manifest = _make_manifest()
        opt = ZeroGrad(manifest, optax.adamw(0.01), population_size=4, rank=2, sigma=0.1, seed=42, run_id="test")
        state = opt.init(params)
        assert state.generation == 0

    def test_step_advances_generation(self):
        params = _make_params()
        manifest = _make_manifest()
        opt = ZeroGrad(manifest, optax.adamw(0.01), population_size=4, rank=2, sigma=0.1, seed=42, run_id="test")
        state = opt.init(params)

        def loss_fn(p, candidate, batch, rng):
            w = p["linear"]["weight"]
            return jnp.sum(w * w), None

        new_params, new_state, metrics = opt.step(state, params, None, loss_fn)
        assert new_state.generation == 1
        assert metrics.population_size == 4
        assert metrics.generation == 0

    def test_step_descends_loss(self):
        """Verify the descent→positive-gradient sign boundary."""
        params = {"w": jnp.ones((4, 4))}
        manifest = Manifest(version=1, entries=(ManifestEntry(("w",), ParameterLayout.MATRIX, "w"),))
        opt = ZeroGrad(
            manifest,
            optax.sgd(learning_rate=0.1),
            population_size=64,
            rank=4,
            sigma=0.05,
            seed=42,
            run_id="test",
        )
        state = opt.init(params)
        x = jnp.ones((1, 4))

        def loss_fn(p, candidate, batch, rng):
            y = candidate.linear(p, ("w",), x)
            return jnp.sum(y ** 2), None

        new_params, new_state, metrics = opt.step(state, params, None, loss_fn)
        loss_before = float(jnp.sum((x @ params["w"]) ** 2))
        loss_after = float(jnp.sum((x @ new_params["w"]) ** 2))
        assert loss_after < loss_before, f"loss did not decrease: {loss_before} -> {loss_after}"

    def test_step_does_not_mutate_inputs(self):
        params = _make_params()
        params_before = jax.tree_util.tree_map(jnp.array, params)
        manifest = _make_manifest()
        opt = ZeroGrad(manifest, optax.adamw(0.01), population_size=4, rank=2, sigma=0.1, seed=42, run_id="test")
        state = opt.init(params)

        def loss_fn(p, candidate, batch, rng):
            return jnp.sum(p["linear"]["weight"] ** 2), None

        opt.step(state, params, None, loss_fn)
        for before, after in zip(
            jax.tree_util.tree_leaves(params_before),
            jax.tree_util.tree_leaves(params),
        ):
            np.testing.assert_array_equal(np.asarray(before), np.asarray(after))

    def test_step_rejects_non_finite_losses(self):
        params = _make_params()
        manifest = _make_manifest()
        opt = ZeroGrad(manifest, optax.adamw(0.01), population_size=4, rank=2, sigma=0.1, seed=42, run_id="test")
        state = opt.init(params)

        def loss_fn(p, candidate, batch, rng):
            return jnp.inf, None

        with pytest.raises(ValueError):
            opt.step(state, params, None, loss_fn)

    def test_non_manifest_params_get_zero_gradient(self):
        params = {
            "linear": {"weight": jnp.ones((8, 4))},
            "frozen": {"weight": jnp.ones((8, 4))},
        }
        manifest = Manifest(
            version=1,
            entries=(ManifestEntry(("linear", "weight"), ParameterLayout.MATRIX, "linear"),),
        )
        opt = ZeroGrad(manifest, optax.adamw(1.0, weight_decay=0.0), population_size=4, rank=2, sigma=0.1, seed=42, run_id="test")
        state = opt.init(params)

        def loss_fn(p, candidate, batch, rng):
            return jnp.sum(p["linear"]["weight"] ** 2), None

        new_params, _, _ = opt.step(state, params, None, loss_fn)
        np.testing.assert_allclose(
            np.asarray(new_params["frozen"]["weight"]),
            np.asarray(params["frozen"]["weight"]),
            rtol=1e-6, atol=1e-6,
        )
