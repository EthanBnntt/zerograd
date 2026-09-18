"""Validation and behavior tests for deterministic PRNG key derivation."""

import jax.numpy as jnp
import pytest

from zerograd import (
    Manifest,
    ManifestEntry,
    ParameterLayout,
    candidate_key,
    group_key,
    step_key,
)


def _make_manifest():
    return Manifest(
        version=1,
        entries=(
            ManifestEntry(("a",), ParameterLayout.VECTOR, "g_a"),
            ManifestEntry(("b",), ParameterLayout.VECTOR, "g_b"),
        ),
    )


class TestStepKey:
    @pytest.mark.parametrize("seed,exc", [
        (1.5, TypeError),
        (True, TypeError),
    ])
    def test_seed_validation(self, seed, exc):
        with pytest.raises(exc):
            step_key(seed, "run", 0, 1)

    @pytest.mark.parametrize("run_id,exc", [
        ("", ValueError),
        (None, ValueError),
    ])
    def test_run_id_validation(self, run_id, exc):
        with pytest.raises(exc):
            step_key(42, run_id, 0, 1)

    @pytest.mark.parametrize("generation,exc", [
        (-1, ValueError),
        (1.5, ValueError),
    ])
    def test_generation_validation(self, generation, exc):
        with pytest.raises(exc):
            step_key(42, "run", generation, 1)

    @pytest.mark.parametrize("version,exc", [
        (0, ValueError),
        (1.5, ValueError),
    ])
    def test_manifest_version_validation(self, version, exc):
        with pytest.raises(exc):
            step_key(42, "run", 0, version)

    def test_key_is_deterministic_for_same_inputs(self):
        k1 = step_key(42, "run", 3, 1)
        k2 = step_key(42, "run", 3, 1)
        assert jnp.array_equal(k1, k2)

    def test_key_differs_for_different_generation(self):
        k1 = step_key(42, "run", 0, 1)
        k2 = step_key(42, "run", 1, 1)
        assert not jnp.array_equal(k1, k2)

    def test_key_differs_for_different_run_id(self):
        k1 = step_key(42, "run-a", 0, 1)
        k2 = step_key(42, "run-b", 0, 1)
        assert not jnp.array_equal(k1, k2)

    def test_key_differs_for_different_seed(self):
        k1 = step_key(42, "run", 0, 1)
        k2 = step_key(43, "run", 0, 1)
        assert not jnp.array_equal(k1, k2)


class TestCandidateKey:
    def test_bool_candidate_rejected(self):
        base = step_key(42, "run", 0, 1)
        with pytest.raises(TypeError):
            candidate_key(base, True)

    def test_negative_int_candidate_rejected(self):
        base = step_key(42, "run", 0, 1)
        with pytest.raises(ValueError):
            candidate_key(base, -1)

    def test_int_and_scalar_array_keys_match(self):
        base = step_key(42, "run", 0, 1)
        k_int = candidate_key(base, 7)
        k_arr = candidate_key(base, jnp.asarray(7))
        assert jnp.array_equal(k_int, k_arr)

    def test_different_candidates_give_different_keys(self):
        base = step_key(42, "run", 0, 1)
        k0 = candidate_key(base, 0)
        k1 = candidate_key(base, 1)
        assert not jnp.array_equal(k0, k1)

    @pytest.mark.parametrize("candidate,exc", [
        (jnp.asarray([0, 1]), TypeError),
        (jnp.asarray(1.5), TypeError),
        (jnp.asarray(-5), ValueError),
    ])
    def test_array_candidate_validation(self, candidate, exc):
        base = step_key(42, "run", 0, 1)
        with pytest.raises(exc):
            candidate_key(base, candidate)


class TestGroupKey:
    def test_group_key_is_stable_and_ordered(self):
        manifest = _make_manifest()
        ck = candidate_key(step_key(42, "run", 0, 1), 0)
        g0 = group_key(ck, manifest, "g_a")
        g1 = group_key(ck, manifest, "g_b")
        assert not jnp.array_equal(g0, g1)

    def test_group_key_unknown_group_raises(self):
        manifest = _make_manifest()
        ck = candidate_key(step_key(42, "run", 0, 1), 0)
        with pytest.raises(KeyError):
            group_key(ck, manifest, "nope")
