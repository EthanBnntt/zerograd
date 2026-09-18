"""Tests for factor-only candidate forward operations and dense reconstruction."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from zerograd import Manifest, ManifestEntry, ParameterLayout, group_key
from zerograd._candidate import (
    perturbed_linear,
    perturbed_table_lookup,
    perturbed_tied_logits,
    perturbed_vector,
)
from zerograd._factors import matrix_factors, scaled_factor, table_factors, vector_noise


def _make_params():
    return {
        "linear": {"weight": jnp.ones((8, 4))},
        "table": {"embed": jnp.ones((16, 4))},
        "vector": {"scale": jnp.ones((4,))},
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


def _manifest():
    return Manifest(version=1, entries=(
        ManifestEntry(("m",), ParameterLayout.MATRIX, "m"),
        ManifestEntry(("t",), ParameterLayout.TABLE, "t"),
        ManifestEntry(("v",), ParameterLayout.VECTOR, "v"),
    ))


class TestCandidateForward:
    def test_linear_matches_dense_reconstruction(self):
        params = _make_params()
        manifest = _make_manifest()
        key = jax.random.key(42)
        rank, sigma = 2, 0.1
        x = jax.random.normal(jax.random.key(1), (3, 8))
        gk = group_key(key, manifest, "linear")
        y = perturbed_linear(x, params["linear"]["weight"], gk, rank, sigma)
        a, b = matrix_factors(gk, (8, 4), rank, dtype=jnp.float32)
        y_dense = x @ params["linear"]["weight"] + scaled_factor(rank, sigma, jnp.float32) * ((x @ a) @ b)
        np.testing.assert_allclose(np.asarray(y), np.asarray(y_dense), rtol=1e-5, atol=1e-5)

    def test_table_lookup_matches_dense(self):
        params = _make_params()
        manifest = _make_manifest()
        key = jax.random.key(42)
        rank, sigma = 2, 0.1
        indices = jnp.array([0, 3, 7])
        gk = group_key(key, manifest, "embed")
        y = perturbed_table_lookup(params["table"]["embed"], indices, gk, rank, sigma)
        a, b = table_factors(gk, (16, 4), rank, dtype=jnp.float32)
        y_dense = params["table"]["embed"][indices] + scaled_factor(rank, sigma, jnp.float32) * jnp.einsum("...r,cr->...c", a[indices], b)
        np.testing.assert_allclose(np.asarray(y), np.asarray(y_dense), rtol=1e-5, atol=1e-5)

    def test_tied_logits_shares_factors_with_lookup(self):
        params = _make_params()
        manifest = _make_manifest()
        key = jax.random.key(42)
        rank, sigma = 2, 0.1
        x = jax.random.normal(jax.random.key(1), (3, 4))
        gk = group_key(key, manifest, "embed")
        logits = perturbed_tied_logits(x, params["table"]["embed"], gk, rank, sigma)
        a, b = table_factors(gk, (16, 4), rank, dtype=jnp.float32)
        y_dense = x @ params["table"]["embed"].T + scaled_factor(rank, sigma, jnp.float32) * jnp.einsum("...c,cr,vr->...v", x, b, a)
        np.testing.assert_allclose(np.asarray(logits), np.asarray(y_dense), rtol=1e-5, atol=1e-5)

    def test_vector_adds_deterministic_noise(self):
        params = _make_params()
        manifest = _make_manifest()
        key = jax.random.key(42)
        sigma = 0.1
        gk = group_key(key, manifest, "scale")
        v = perturbed_vector(params["vector"]["scale"], gk, sigma)
        noise = vector_noise(gk, (4,), dtype=jnp.float32)
        v_dense = params["vector"]["scale"] + scaled_factor(1, sigma, jnp.float32) * noise
        np.testing.assert_allclose(np.asarray(v), np.asarray(v_dense), rtol=1e-5, atol=1e-5)


class TestForwardShapeValidation:
    def test_perturbed_linear_rejects_incompatible_shapes(self):
        with pytest.raises(ValueError):
            perturbed_linear(jnp.ones((3, 5)), jnp.ones((8, 4)), jax.random.key(0), 2, 0.1)

    def test_perturbed_linear_rejects_non_2d_weight(self):
        with pytest.raises(ValueError):
            perturbed_linear(jnp.ones((3, 8)), jnp.ones((4,)), jax.random.key(0), 2, 0.1)

    def test_perturbed_table_lookup_rejects_non_2d_table(self):
        with pytest.raises(ValueError):
            perturbed_table_lookup(jnp.ones((16,)), jnp.array([0, 1]), jax.random.key(0), 2, 0.1)

    def test_perturbed_tied_logits_rejects_incompatible_shapes(self):
        with pytest.raises(ValueError):
            perturbed_tied_logits(jnp.ones((3, 5)), jnp.ones((16, 4)), jax.random.key(0), 2, 0.1)

    def test_perturbed_tied_logits_rejects_non_2d_table(self):
        with pytest.raises(ValueError):
            perturbed_tied_logits(jnp.ones((3, 4)), jnp.ones((16,)), jax.random.key(0), 2, 0.1)

    def test_perturbed_vector_rejects_non_1d_vector(self):
        with pytest.raises(ValueError):
            perturbed_vector(jnp.ones((4, 4)), jax.random.key(0), 0.1)


class TestPerturbedForwardResults:
    def test_linear_preserves_output_shape(self):
        y = perturbed_linear(
            jnp.ones((3, 8)), jnp.ones((8, 4)), jax.random.key(1), 2, 0.1
        )
        assert y.shape == (3, 4)

    def test_table_lookup_preserves_output_shape(self):
        indices = jnp.array([0, 3, 7, 15])
        y = perturbed_table_lookup(jnp.ones((16, 4)), indices, jax.random.key(1), 2, 0.1)
        assert y.shape == (4, 4)

    def test_tied_logits_preserves_output_shape(self):
        y = perturbed_tied_logits(
            jnp.ones((5, 4)), jnp.ones((16, 4)), jax.random.key(1), 2, 0.1
        )
        assert y.shape == (5, 16)

    def test_vector_preserves_output_shape(self):
        v = perturbed_vector(jnp.ones((4,)), jax.random.key(1), 0.1)
        assert v.shape == (4,)
