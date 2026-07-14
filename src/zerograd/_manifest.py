"""Explicit parameter identity and layout metadata for ZeroGrad."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias

import jax


ParameterPath: TypeAlias = tuple[str, ...]
ParameterTree: TypeAlias = Mapping[str, "ParameterTree | jax.Array"]


class ParameterLayout(StrEnum):
    """Factor algebra selected for a manifest parameter leaf."""

    MATRIX = "matrix"
    TABLE = "table"
    VECTOR = "vector"


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    """One selected parameter leaf and its deterministic perturbation group."""

    path: ParameterPath
    layout: ParameterLayout
    group: str

    def __post_init__(self) -> None:
        if not isinstance(self.path, tuple) or not self.path:
            raise ValueError("manifest paths must be non-empty tuples")
        if any(not isinstance(part, str) or not part for part in self.path):
            raise ValueError("manifest path components must be non-empty strings")
        if not isinstance(self.layout, ParameterLayout):
            raise TypeError("manifest layouts must be ParameterLayout values")
        if not isinstance(self.group, str) or not self.group:
            raise ValueError("manifest groups must be non-empty strings")


@dataclass(frozen=True, slots=True)
class Manifest:
    """Versioned, ordered manifest for explicit perturbation replay identity."""

    version: int
    entries: tuple[ManifestEntry, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.version, int)
            or isinstance(self.version, bool)
            or self.version < 1
        ):
            raise ValueError(
                f"manifest version must be a positive integer, got {self.version!r}"
            )
        if not isinstance(self.entries, tuple) or not self.entries:
            raise ValueError("manifest entries must be a non-empty tuple")
        if any(not isinstance(entry, ManifestEntry) for entry in self.entries):
            raise TypeError("manifest entries must be ManifestEntry instances")
        paths = tuple(entry.path for entry in self.entries)
        if len(set(paths)) != len(paths):
            raise ValueError("manifest paths must be unique")
        groups = tuple(entry.group for entry in self.entries)
        if len(set(groups)) != len(groups):
            raise ValueError("manifest groups must be unique; reuse one entry for tied uses")

    def entry(self, path: ParameterPath) -> ManifestEntry:
        """Return the explicitly registered entry for ``path``."""
        for entry in self.entries:
            if entry.path == path:
                return entry
        raise KeyError(f"path is not in the manifest: {'.'.join(path)}")

    def group_index(self, group: str) -> int:
        """Return the canonical split index for one manifest group."""
        for index, entry in enumerate(self.entries):
            if entry.group == group:
                return index
        raise KeyError(f"group is not in the manifest: {group}")

    def resolve(self, params: ParameterTree, path: ParameterPath) -> jax.Array:
        """Resolve an explicitly named array leaf without relying on tree order."""
        node: ParameterTree | jax.Array = params
        for part in path:
            if not isinstance(node, Mapping):
                raise KeyError(
                    f"path traverses an array before {part!r}: {'.'.join(path)}"
                )
            try:
                node = node[part]
            except KeyError as error:
                raise KeyError(f"parameter path is missing: {'.'.join(path)}") from error
        if not isinstance(node, jax.Array):
            raise TypeError(
                f"parameter path does not resolve to a JAX array: {'.'.join(path)}"
            )
        return node

    def validate(self, params: ParameterTree) -> None:
        """Check all selected paths and layout dimensionalities before a step."""
        for entry in self.entries:
            parameter = self.resolve(params, entry.path)
            expected_ndim = 1 if entry.layout is ParameterLayout.VECTOR else 2
            if parameter.ndim != expected_ndim:
                raise ValueError(
                    f"{entry.layout.value} parameter {'.'.join(entry.path)} must be "
                    f"{expected_ndim}-D, got shape {parameter.shape}"
                )
