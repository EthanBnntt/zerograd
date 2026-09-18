"""Validation tests for ManifestEntry and Manifest metadata + resolution."""

from typing import Any

import jax
import jax.numpy as jnp
import pytest

from zerograd import Manifest, ManifestEntry, ParameterLayout


def _entry(path, layout=ParameterLayout.VECTOR, group=None):
    return ManifestEntry(path, layout, group or ("g_" + "_".join(path)))


class TestManifestEntryValidation:
    def test_rejects_non_tuple_path(self):
        bad_path: Any = "not-a-tuple"
        with pytest.raises(ValueError):
            ManifestEntry(bad_path, ParameterLayout.VECTOR, "g")

    def test_rejects_empty_path(self):
        with pytest.raises(ValueError):
            ManifestEntry((), ParameterLayout.VECTOR, "g")

    def test_rejects_non_string_path_component(self):
        bad_path: Any = ("a", 1)
        with pytest.raises(ValueError):
            ManifestEntry(bad_path, ParameterLayout.VECTOR, "g")

    def test_rejects_empty_string_path_component(self):
        with pytest.raises(ValueError):
            ManifestEntry(("a", ""), ParameterLayout.VECTOR, "g")

    def test_rejects_non_layout_value(self):
        bad_layout: Any = "not_a_layout"
        with pytest.raises(TypeError):
            ManifestEntry(("a",), bad_layout, "g")

    def test_rejects_empty_group(self):
        with pytest.raises(ValueError):
            ManifestEntry(("a",), ParameterLayout.VECTOR, "")

    def test_rejects_non_string_group(self):
        bad_group: Any = 5
        with pytest.raises(ValueError):
            ManifestEntry(("a",), ParameterLayout.VECTOR, bad_group)


class TestManifestValidation:
    def test_rejects_bool_version(self):
        bad_version: Any = True
        with pytest.raises(ValueError):
            Manifest(version=bad_version, entries=(_entry(("a",)),))

    def test_rejects_non_positive_version(self):
        with pytest.raises(ValueError):
            Manifest(version=0, entries=(_entry(("a",)),))

    def test_rejects_non_tuple_entries(self):
        bad_entries: Any = [_entry(("a",))]
        with pytest.raises(ValueError):
            Manifest(version=1, entries=bad_entries)

    def test_rejects_empty_entries(self):
        with pytest.raises(ValueError):
            Manifest(version=1, entries=())

    def test_rejects_non_manifest_entry(self):
        bad_entries: Any = (_entry(("a",)), "not-an-entry")
        with pytest.raises(TypeError):
            Manifest(version=1, entries=bad_entries)

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

    def test_accepts_valid_manifest(self):
        m = Manifest(version=1, entries=(_entry(("a",)), _entry(("b",))))
        assert len(m.entries) == 2

    def test_equality_ignores_internal_cache(self):
        m1 = Manifest(version=1, entries=(
            ManifestEntry(("w",), ParameterLayout.MATRIX, "w"),
            ManifestEntry(("v",), ParameterLayout.VECTOR, "v"),
        ))
        m2 = Manifest(version=1, entries=(
            ManifestEntry(("w",), ParameterLayout.MATRIX, "w"),
            ManifestEntry(("v",), ParameterLayout.VECTOR, "v"),
        ))
        assert m1 == m2
        assert hash(m1) == hash(m2)


class TestManifestResolveAndLookup:
    def _make(self):
        return Manifest(version=2, entries=(
            ManifestEntry(("layer", "weight"), ParameterLayout.MATRIX, "lw"),
            ManifestEntry(("layer", "bias"), ParameterLayout.VECTOR, "lb"),
        ))

    def test_entry_returns_registered_entry(self):
        m = self._make()
        entry = m.entry(("layer", "weight"))
        assert entry.layout is ParameterLayout.MATRIX

    def test_entry_unknown_path_raises(self):
        m = self._make()
        with pytest.raises(KeyError):
            m.entry(("nope",))

    def test_group_index_returns_position(self):
        m = self._make()
        assert m.group_index("lw") == 0
        assert m.group_index("lb") == 1

    def test_group_index_unknown_raises(self):
        m = self._make()
        with pytest.raises(KeyError):
            m.group_index("zzz")

    def test_resolve_returns_array_leaf(self):
        m = self._make()
        params = {"layer": {"weight": jnp.ones((4, 2)), "bias": jnp.zeros((2,))}}
        result = m.resolve(params, ("layer", "bias"))
        assert isinstance(result, jax.Array)
        assert result.shape == (2,)

    def test_resolve_missing_path_raises(self):
        m = self._make()
        with pytest.raises(KeyError):
            m.resolve({"layer": {"weight": jnp.ones((4, 2))}}, ("layer", "bias"))

    def test_resolve_traverses_array_before_end_raises(self):
        m = self._make()
        with pytest.raises(KeyError):
            m.resolve({"layer": {"weight": jnp.ones((4, 2))}}, ("layer", "weight", "extra"))

    def test_resolve_non_array_leaf_raises(self):
        m = self._make()
        with pytest.raises(TypeError):
            m.resolve({"layer": {"weight": jnp.ones((4, 2))}}, ("layer",))


class TestManifestValidate:
    def test_validate_accepts_correct_dimensionality(self):
        m = Manifest(version=1, entries=(
            ManifestEntry(("w",), ParameterLayout.MATRIX, "w"),
            ManifestEntry(("v",), ParameterLayout.VECTOR, "v"),
        ))
        m.validate({"w": jnp.ones((4, 2)), "v": jnp.ones((2,))})

    def test_validate_rejects_vector_with_wrong_ndim(self):
        m = Manifest(version=1, entries=(ManifestEntry(("v",), ParameterLayout.VECTOR, "v"),))
        with pytest.raises(ValueError):
            m.validate({"v": jnp.ones((2, 2))})

    def test_validate_rejects_matrix_with_wrong_ndim(self):
        m = Manifest(version=1, entries=(ManifestEntry(("w",), ParameterLayout.MATRIX, "w"),))
        with pytest.raises(ValueError):
            m.validate({"w": jnp.ones((4,))})
