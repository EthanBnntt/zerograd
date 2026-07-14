"""Validation tests for factor-only candidate forward operations."""

import jax
import jax.numpy as jnp
import pytest

from zerograd import CandidateContext, Manifest, ManifestEntry, ParameterLayout
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


def _params():
    return {
        "m": jnp.ones((8, 4)),
        "t": jnp.ones((16, 4)),
        "v": jnp.ones((4,)),
    }


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


class TestCandidateContextLayoutMismatch:
    def _ctx(self):
        return CandidateContext(_manifest(), jax.random.key(0), 2, 0.1)

    def test_linear_rejects_non_matrix_entry(self):
        ctx = self._ctx()
        with pytest.raises(ValueError):
            ctx.linear(_params(), ("t",), jnp.ones((3, 16)))

    def test_table_lookup_rejects_non_table_entry(self):
        ctx = self._ctx()
        with pytest.raises(ValueError):
            ctx.table_lookup(_params(), ("m",), jnp.array([0, 1]))

    def test_tied_logits_rejects_non_table_entry(self):
        ctx = self._ctx()
        with pytest.raises(ValueError):
            ctx.tied_logits(_params(), ("m",), jnp.ones((3, 4)))

    def test_vector_rejects_non_vector_entry(self):
        ctx = self._ctx()
        with pytest.raises(ValueError):
            ctx.vector(_params(), ("t",))


class TestCandidateContextForwardResults:
    def test_linear_preserves_output_shape(self):
        ctx = CandidateContext(_manifest(), jax.random.key(1), 2, 0.1)
        x = jnp.ones((3, 8))
        y = ctx.linear(_params(), ("m",), x)
        assert y.shape == (3, 4)

    def test_table_lookup_preserves_output_shape(self):
        ctx = CandidateContext(_manifest(), jax.random.key(1), 2, 0.1)
        indices = jnp.array([0, 3, 7, 15])
        y = ctx.table_lookup(_params(), ("t",), indices)
        assert y.shape == (4, 4)

    def test_tied_logits_preserves_output_shape(self):
        ctx = CandidateContext(_manifest(), jax.random.key(1), 2, 0.1)
        x = jnp.ones((5, 4))
        y = ctx.tied_logits(_params(), ("t",), x)
        assert y.shape == (5, 16)

    def test_vector_preserves_output_shape(self):
        ctx = CandidateContext(_manifest(), jax.random.key(1), 1, 0.1)
        v = ctx.vector(_params(), ("v",))
        assert v.shape == (4,)

    def test_key_for_is_deterministic(self):
        manifest = _manifest()
        ctx = CandidateContext(manifest, jax.random.key(7), 2, 0.1)
        k1 = ctx.key_for(("m",))
        k2 = ctx.key_for(("m",))
        assert jnp.array_equal(k1, k2)
        # Different groups produce different keys.
        k3 = ctx.key_for(("t",))
        assert not jnp.array_equal(k1, k3)
