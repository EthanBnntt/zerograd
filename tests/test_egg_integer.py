"""EGG-style int8 clip-cast / scaled matmul helpers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from zerograd._integer import (
    EGG_I8_MAX,
    EGG_I8_MIN,
    egg_clip_cast,
    egg_init_matrix,
    egg_matmul_divisor,
    egg_requantize,
)


class TestEggInteger:
    def test_clip_cast_saturates(self):
        x = jnp.array([-1000, -127, 0, 127, 500], dtype=jnp.int32)
        y = egg_clip_cast(x)
        assert y.dtype == jnp.int8
        np.testing.assert_array_equal(
            y, np.array([EGG_I8_MIN, -127, 0, 127, EGG_I8_MAX], dtype=np.int8)
        )

    def test_matmul_divisor_matches_paper(self):
        # 16 * sqrt(n)
        assert egg_matmul_divisor(1) == 16
        assert egg_matmul_divisor(64) == 128  # 16*8
        assert egg_matmul_divisor(256) == 256  # 16*16

    def test_egg_requantize_uses_divisor(self):
        # accum = 256 * 128 → //128 = 256 → clip to 127
        y = jnp.full((2, 4), 256 * 128, dtype=jnp.int32)
        out = egg_requantize(y, in_features=64)
        assert out.dtype == jnp.int8
        assert int(out[0, 0]) == EGG_I8_MAX

    def test_egg_init_in_range(self):
        w = egg_init_matrix(jax.random.key(0), (64, 32))
        assert w.dtype == jnp.int8
        assert int(w.min()) >= EGG_I8_MIN
        assert int(w.max()) <= EGG_I8_MAX
