"""Tests for manifest construction, candidate forward helpers, and replay parity."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from zerograd import (
    CandidateContext,
    Manifest,
    ManifestEntry,
    ParameterLayout,
    candidate_key,
    group_key,
    matrix_factors,
    shape_centered_loss,
    step_key,
    validate_losses,
)


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


class TestManifest:
    def test_valid_manifest_accepts_correct_dimensions(self):
        manifest = _make_manifest()
        manifest.validate(_make_params())

    def test_rejects_non_positive_version(self):
        with pytest.raises(ValueError):
            Manifest(version=0, entries=(ManifestEntry(("a",), ParameterLayout.VECTOR, "g"),))

    def test_rejects_duplicate_paths(self):
        with pytest.raises(ValueError):
            Manifest(
                version=1,
                entries=(
                    ManifestEntry(("a",), ParameterLayout.VECTOR, "g1"),
                    ManifestEntry(("a",), ParameterLayout.VECTOR, "g2"),
                ),
            )

    def test_rejects_duplicate_groups(self):
        with pytest.raises(ValueError):
            Manifest(
                version=1,
                entries=(
                    ManifestEntry(("a",), ParameterLayout.VECTOR, "g"),
                    ManifestEntry(("b",), ParameterLayout.VECTOR, "g"),
                ),
            )

    def test_rejects_non_parameter_layout(self):
        with pytest.raises(TypeError):
            ManifestEntry(("a",), "not_a_layout", "g")

    def test_rejects_wrong_dimensionality(self):
        manifest = _make_manifest()
        bad_params = {
            "linear": {"weight": jnp.ones((4,))},
            "table": {"embed": jnp.ones((16, 4))},
            "vector": {"scale": jnp.ones((4,))},
        }
        with pytest.raises(ValueError):
            manifest.validate(bad_params)

    def test_group_index_follows_entry_order(self):
        manifest = _make_manifest()
        assert manifest.group_index("linear") == 0
        assert manifest.group_index("embed") == 1
        assert manifest.group_index("scale") == 2


class TestCandidateForward:
    def test_linear_matches_dense_reconstruction(self):
        params = _make_params()
        manifest = _make_manifest()
        key = jax.random.key(42)
        rank, sigma = 2, 0.1
        x = jax.random.normal(jax.random.key(1), (3, 8))
        ctx = CandidateContext(manifest, key, rank, sigma)
        y = ctx.linear(params, ("linear", "weight"), x)
        from zerograd._factors import matrix_factors, scaled_factor
        gk = group_key(key, manifest, "linear")
        a, b = matrix_factors(gk, (8, 4), rank, dtype=jnp.float32)
        y_dense = x @ params["linear"]["weight"] + scaled_factor(rank, sigma, jnp.float32) * ((x @ a) @ b)
        np.testing.assert_allclose(np.asarray(y), np.asarray(y_dense), rtol=1e-5, atol=1e-5)

    def test_table_lookup_matches_dense(self):
        params = _make_params()
        manifest = _make_manifest()
        key = jax.random.key(42)
        rank, sigma = 2, 0.1
        indices = jnp.array([0, 3, 7])
        ctx = CandidateContext(manifest, key, rank, sigma)
        y = ctx.table_lookup(params, ("table", "embed"), indices)
        from zerograd._factors import table_factors, scaled_factor
        gk = group_key(key, manifest, "embed")
        a, b = table_factors(gk, (16, 4), rank, dtype=jnp.float32)
        y_dense = params["table"]["embed"][indices] + scaled_factor(rank, sigma, jnp.float32) * jnp.einsum("...r,cr->...c", a[indices], b)
        np.testing.assert_allclose(np.asarray(y), np.asarray(y_dense), rtol=1e-5, atol=1e-5)

    def test_tied_logits_shares_factors_with_lookup(self):
        params = _make_params()
        manifest = _make_manifest()
        key = jax.random.key(42)
        rank, sigma = 2, 0.1
        x = jax.random.normal(jax.random.key(1), (3, 4))
        ctx = CandidateContext(manifest, key, rank, sigma)
        logits = ctx.tied_logits(params, ("table", "embed"), x)
        from zerograd._factors import table_factors, scaled_factor
        gk = group_key(key, manifest, "embed")
        a, b = table_factors(gk, (16, 4), rank, dtype=jnp.float32)
        y_dense = x @ params["table"]["embed"].T + scaled_factor(rank, sigma, jnp.float32) * jnp.einsum("...c,cr,vr->...v", x, b, a)
        np.testing.assert_allclose(np.asarray(logits), np.asarray(y_dense), rtol=1e-5, atol=1e-5)

    def test_vector_adds_deterministic_noise(self):
        params = _make_params()
        manifest = _make_manifest()
        key = jax.random.key(42)
        sigma = 0.1
        ctx = CandidateContext(manifest, key, 1, sigma)
        v = ctx.vector(params, ("vector", "scale"))
        from zerograd._factors import vector_noise, scaled_factor
        gk = group_key(key, manifest, "scale")
        noise = vector_noise(gk, (4,), dtype=jnp.float32)
        v_dense = params["vector"]["scale"] + scaled_factor(1, sigma, jnp.float32) * noise
        np.testing.assert_allclose(np.asarray(v), np.asarray(v_dense), rtol=1e-5, atol=1e-5)

    def test_layout_mismatch_rejected(self):
        params = _make_params()
        manifest = _make_manifest()
        ctx = CandidateContext(manifest, jax.random.key(0), 2, 0.1)
        with pytest.raises(ValueError):
            ctx.linear(params, ("table", "embed"), jnp.ones((3, 16)))


class TestFitness:
    def test_shape_centered_loss_formula(self):
        losses = jnp.array([1.0, 2.0, 3.0, 4.0])
        sigma = 0.1
        shaped = shape_centered_loss(losses, sigma)
        expected = -(losses - jnp.mean(losses)) / (4 * sigma)
        np.testing.assert_allclose(np.asarray(shaped), np.asarray(expected), rtol=1e-6)

    def test_validate_losses_rejects_non_finite(self):
        with pytest.raises(ValueError):
            validate_losses(jnp.array([1.0, jnp.inf, 3.0, 4.0]))

    def test_validate_losses_rejects_too_few(self):
        with pytest.raises(ValueError):
            validate_losses(jnp.array([1.0]))
