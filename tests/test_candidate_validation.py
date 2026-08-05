"""Validation tests for factor-only candidate forward operations."""

import jax
import jax.numpy as jnp
import pytest

from zerograd import group_key, Manifest, ManifestEntry, ParameterLayout
from zerograd._candidate import (
    perturbed_linear,
    perturbed_table_lookup,
    perturbed_tied_logits,
    perturbed_vector,
)


def _manifest():
    return Manifest(version=1, entries=(
        ManifestEntry(("m",), ParameterLayout.MATRIX, "m"),
        ManifestEntry(("t",), ParameterLayout.TABLE, "t"),
        ManifestEntry(("v",), ParameterLayout.VECTOR, "v"),
    ))


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
        # x last-dim (5) != table cols (4)
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

    def test_group_key_is_deterministic(self):
        manifest = _manifest()
        key = jax.random.key(7)
        k1 = group_key(key, manifest, "m")
        k2 = group_key(key, manifest, "m")
        assert jnp.array_equal(k1, k2)
        k3 = group_key(key, manifest, "t")
        assert not jnp.array_equal(k1, k3)
