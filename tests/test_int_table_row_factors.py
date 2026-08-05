"""Bulk int table factors + gathered-row helper stay consistent."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from zerograd._candidate import perturbed_int_table_lookup
from zerograd._factors import int_table_factors, int_table_factors_for_rows


class TestIntTableRowFactors:
    def test_sparse_rows_match_full_table_index(self):
        key = jax.random.key(0)
        rows, cols, rank = 64, 8, 2
        a_full, b_full = int_table_factors(key, (rows, cols), rank)
        idx = jnp.asarray([0, 3, 17, 63, 17], dtype=jnp.int32)
        a_rows, b_rows = int_table_factors_for_rows(key, (rows, cols), rank, idx)
        np.testing.assert_array_equal(np.asarray(a_rows), np.asarray(a_full[idx]))
        np.testing.assert_array_equal(np.asarray(b_rows), np.asarray(b_full))

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
