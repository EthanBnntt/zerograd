"""Bitwise correctness: fused_linear_lut (Pallas/Triton) vs unfused reference.

GPU-only tests — they skip on CPU/ROCm. The reference is
``linear_lut_reference`` which mirrors ``IntLinearLUT.__call__`` exactly
(int8 @ int8 -> int32, floor-div by round(16*sqrt(K)), clip ±127, LUT gather).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from zerograd import IntLinearLUT, fused_linear_lut, int_linear_lut_fused
from zerograd._fused_lut import linear_lut_reference

GPU = jax.default_backend() == "gpu"
GPU_ONLY = pytest.mark.skipif(not GPU, reason="fused kernel requires CUDA GPU")

SHAPES = [  # (M, K, N) — model-typical plus odd M for the padding path
    (64, 128, 128),
    (256, 256, 256),
    (512, 256, 1536),  # 6D in-proj
    (1024, 512, 512),
    (130, 256, 256),  # M not a multiple of block_m
    (64, 1024, 1024),
    (2048, 2048, 512),  # deep-K FFN-like
]


def _rand_i8(key, shape):
    return jax.random.randint(key, shape, -128, 128, jnp.int8)


def _luts(key):
    domain = jnp.arange(-128, 128, dtype=jnp.int32)
    return {
        "identity": domain.astype(jnp.int8),
        "relu": jnp.maximum(domain, 0).astype(jnp.int8),
        "clip_egg": jnp.clip(domain, -127, 127).astype(jnp.int8),
        "random": _rand_i8(key, (256,)),
        "const_min": jnp.full((256,), -127, jnp.int8),
        "const_max": jnp.full((256,), 127, jnp.int8),
    }


@GPU_ONLY
@pytest.mark.parametrize("m,k,n", SHAPES)
def test_fused_matches_reference_random(m, k, n):
    key = jax.random.PRNGKey(m * 7 + k * 13 + n)
    kx, kw, kt = jax.random.split(key, 3)
    x = _rand_i8(kx, (m, k))
    w = _rand_i8(kw, (k, n))
    for name, table in _luts(kt).items():
        got = fused_linear_lut(x, w, table, mode="fused")
        want = linear_lut_reference(x, w, table)
        assert got.dtype == jnp.int8
        assert bool(jnp.all(got == want)), f"LUT {name}: mismatch"


@GPU_ONLY
def test_fused_with_bias():
    key = jax.random.PRNGKey(0)
    kx, kw, kb, kt = jax.random.split(key, 4)
    x = _rand_i8(kx, (256, 256))
    w = _rand_i8(kw, (256, 256))
    bias = jax.random.randint(kb, (256,), -(2**20), 2**20, jnp.int32)
    table = _rand_i8(kt, (256,))
    got = fused_linear_lut(x, w, table, bias=bias, mode="fused")
    want = linear_lut_reference(x, w, table, bias=bias)
    assert bool(jnp.all(got == want))


@GPU_ONLY
def test_fused_extreme_activations():
    # All ±127 inputs: max-magnitude accumulators, exercises floor-div on
    # large negatives and saturation.
    key = jax.random.PRNGKey(1)
    w = _rand_i8(key, (128, 128))
    table = _rand_i8(jax.random.PRNGKey(2), (256,))
    for val in (-128, -127, 127, 0):
        x = jnp.full((64, 128), val, jnp.int8)
        got = fused_linear_lut(x, w, table, mode="fused")
        want = linear_lut_reference(x, w, table)
        assert bool(jnp.all(got == want)), f"x={val} mismatch"


@GPU_ONLY
def test_fused_3d_input():
    key = jax.random.PRNGKey(3)
    kx, kw, kt = jax.random.split(key, 3)
    x = _rand_i8(kx, (8, 33, 256))  # (B, T, K) — odd T exercises M padding
    w = _rand_i8(kw, (256, 256))
    table = _rand_i8(kt, (256,))
    got = fused_linear_lut(x, w, table, mode="fused")
    want = linear_lut_reference(x, w, table)
    assert got.shape == (8, 33, 256)
    assert bool(jnp.all(got == want))


@GPU_ONLY
def test_module_drop_in_equivalence():
    # int_linear_lut_fused(IntLinearLUT) == IntLinearLUT(x), incl. bias path.
    for use_bias in (False, True):
        mod = IntLinearLUT(256, 256, use_bias=use_bias, rngs=nnx.Rngs(0))
        mod.lut.table = nnx.Param(_rand_i8(jax.random.PRNGKey(4), (256,)))
        if use_bias:
            mod.linear.bias = nnx.Param(
                jax.random.randint(jax.random.PRNGKey(5), (256,), -(2**18), 2**18, jnp.int32)
            )
        x = _rand_i8(jax.random.PRNGKey(6), (4, 16, 256))
        got = int_linear_lut_fused(mod, x)
        want = mod(x)
        assert bool(jnp.all(got == want)), f"use_bias={use_bias} mismatch"


@GPU_ONLY
def test_no_fallback_on_gpu():
    x = _rand_i8(jax.random.PRNGKey(7), (64, 128))
    w = _rand_i8(jax.random.PRNGKey(8), (128, 128))
    table = _rand_i8(jax.random.PRNGKey(9), (256,))
    got = fused_linear_lut(x, w, table, mode="fused", allow_fallback=False)
    want = linear_lut_reference(x, w, table)
    assert bool(jnp.all(got == want))


# --- validation errors (run everywhere, no GPU needed) ---


def test_validation_errors():
    x = jnp.zeros((4, 128), jnp.int8)
    w = jnp.zeros((128, 64), jnp.int8)
    table = jnp.zeros((256,), jnp.int8)
    with pytest.raises(TypeError):
        fused_linear_lut(x.astype(jnp.float32), w, table)
    with pytest.raises(ValueError, match="256"):
        fused_linear_lut(x, w, table[:128])
    with pytest.raises(ValueError, match="power of two"):
        fused_linear_lut(jnp.zeros((4, 96), jnp.int8), jnp.zeros((96, 64), jnp.int8), table)
    with pytest.raises(ValueError, match="multiple of block_n"):
        fused_linear_lut(x, jnp.zeros((128, 100), jnp.int8), table)
    with pytest.raises(ValueError, match="inner dim"):
        fused_linear_lut(x, jnp.zeros((64, 64), jnp.int8), table)
    # CPU fallback equals reference
    got = fused_linear_lut(x, w, table, mode="fused")
    assert np.array_equal(np.asarray(got), np.asarray(linear_lut_reference(x, w, table)))
