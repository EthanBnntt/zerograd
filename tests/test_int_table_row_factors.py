"""Bulk int table factors stay consistent with perturbed table lookups."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from zerograd._candidate import perturbed_int_table_lookup
from zerograd._factors import int_table_factors


class TestIntTableFactors:
    def test_perturbed_lookup_matches_indexed_factors(self):
        key = jax.random.key(1)
        table = jax.random.randint(key, (32, 6), -40, 40, dtype=jnp.int8)
        idx = jnp.asarray([[1, 2, 3], [30, 0, 15]], dtype=jnp.int32)
        fk = jax.random.key(2)
        y = perturbed_int_table_lookup(table, idx, fk, rank=2, sigma_shift=2)
        a, b = int_table_factors(fk, table.shape, 2)
        base = table[idx].astype(jnp.int32)
        pert = jnp.einsum("...r,cr->...c", a[idx].astype(jnp.int32), b.astype(jnp.int32))
        ref = jnp.clip(base + (pert >> 6), -128, 127).astype(jnp.int8)
        np.testing.assert_array_equal(np.asarray(y), np.asarray(ref))
