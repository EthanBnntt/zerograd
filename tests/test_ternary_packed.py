"""Packed-ternary layer: pack/unpack, reference vs module, fused kernel, training.

The fused Pallas/Triton kernel tests are GPU-only (they run on CUDA/H100 and on
this ROCm stack, which bundles the pallas-triton lowering). The pack/unpack,
reference, surgery, and optimizer tests run everywhere.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from zerograd import (
    IntLinear,
    TernaryLinearLUT,
    ZeroGrad,
    apply_surgery,
    fused_ternary_lut,
    pack_ternary,
    packed_ternary_lut_fused,
    ternary_lut_reference,
    unpack_ternary,
)
from zerograd._integer import TERNARY_PER_BYTE, ternary_init, ternary_requantize
from zerograd._nnx import ZgIntLUT, ZgTernaryIntLinear, params_pure_dict

GPU = jax.default_backend() == "gpu"
GPU_ONLY = pytest.mark.skipif(not GPU, reason="fused kernel requires a GPU backend")


def _rand_i8(key, shape):
    return jax.random.randint(key, shape, -128, 128, jnp.int8)


def _ternary(key, shape):
    return jax.random.randint(key, shape, -1, 2, jnp.int8)


class TestPackUnpack:
    def test_roundtrip(self):
        w = _ternary(jax.random.key(0), (64, 32))
        p = pack_ternary(w)
        assert p.dtype == jnp.uint8
        assert p.shape == (64 // TERNARY_PER_BYTE, 32)
        w2 = unpack_ternary(p)
        assert w2.dtype == jnp.int8
        assert bool(jnp.all(w2 == w))

    def test_byte_encoding(self):
        # Column 0 = [-1, 0, +1, +1] → codes [0,1,2,2] → byte 0|1<<2|2<<4|2<<6.
        w = jnp.array([[-1], [0], [1], [1]], dtype=jnp.int8)
        p = pack_ternary(w)
        assert int(p[0, 0]) == (0 | (1 << 2) | (2 << 4) | (2 << 6))

    def test_values_stay_ternary(self):
        w_t, _ = ternary_init(jax.random.key(1), (128, 64))
        w2 = unpack_ternary(pack_ternary(w_t))
        assert set(np.unique(np.asarray(w2)).tolist()).issubset({-1, 0, 1})

    def test_rejects_non_int8(self):
        with pytest.raises(TypeError):
            pack_ternary(jnp.zeros((8, 4), dtype=jnp.float32))

    def test_rejects_k_not_divisible(self):
        with pytest.raises(ValueError):
            pack_ternary(jnp.zeros((6, 4), dtype=jnp.int8))

    def test_rejects_non_2d(self):
        with pytest.raises(ValueError):
            pack_ternary(jnp.zeros((8,), dtype=jnp.int8))

    def test_unpack_rejects_non_uint8(self):
        with pytest.raises(TypeError):
            unpack_ternary(jnp.zeros((4, 4), dtype=jnp.int8))


class TestReference:
    def test_matches_int_linear_path(self):
        # ternary_lut_reference == IntLinear(int8 ternary kernel) + LUT gather.
        key = jax.random.key(0)
        kx, kw, kt = jax.random.split(key, 3)
        k, n = 256, 128
        x = _rand_i8(kx, (16, k))
        w = _ternary(kw, (k, n))
        table = _rand_i8(kt, (256,))
        packed = pack_ternary(w)
        got = ternary_lut_reference(x, packed, table)
        # Manual ground truth.
        acc = (x.astype(jnp.int32) @ w.astype(jnp.int32))
        req = ternary_requantize(acc, k, act_dtype=jnp.int8)
        want = table[req.astype(jnp.int32) + 128]
        assert bool(jnp.all(got == want))

    def test_bias(self):
        k, n = 64, 64
        x = _rand_i8(jax.random.key(0), (8, k))
        w = _ternary(jax.random.key(1), (k, n))
        bias = jax.random.randint(jax.random.key(2), (n,), -(2**18), 2**18, jnp.int32)
        table = _rand_i8(jax.random.key(3), (256,))
        packed = pack_ternary(w)
        nb = ternary_lut_reference(x, packed, table)
        wb = ternary_lut_reference(x, packed, table, bias=bias)
        assert nb.shape == wb.shape == (8, n)


class TestModule:
    def test_kernel_is_ternary_and_marked(self):
        mod = TernaryLinearLUT(64, 32, rngs=nnx.Rngs(0))
        w = mod.linear.kernel[...]
        assert w.dtype == jnp.int8
        assert set(np.unique(np.asarray(w)).tolist()).issubset({-1, 0, 1})
        from zerograd._nnx import INT_BITS_METADATA_KEY

        assert mod.linear.kernel.get_metadata(INT_BITS_METADATA_KEY) == 2

    def test_call_matches_reference(self):
        mod = TernaryLinearLUT(256, 128, use_bias=True, rngs=nnx.Rngs(0))
        x = _rand_i8(jax.random.key(1), (4, 256))
        packed = pack_ternary(mod.linear.kernel[...])
        want = ternary_lut_reference(
            x, packed, mod.lut.table[...], bias=mod.linear.bias[...]
        )
        assert bool(jnp.all(mod(x) == want))

    def test_surgery_wraps_sublayers(self):
        mod = TernaryLinearLUT(64, 64, rngs=nnx.Rngs(0))
        mod, manifest = apply_surgery(mod, rank=2, sigma=0.05, integer_es=True)
        assert isinstance(mod.linear, ZgTernaryIntLinear)
        assert isinstance(mod.lut, ZgIntLUT)
        paths = {e.path for e in manifest.entries}
        assert ("linear", "kernel") in paths
        assert ("lut", "table") in paths

    def test_eval_unperturbed(self):
        mod = TernaryLinearLUT(64, 64, use_bias=True, rngs=nnx.Rngs(0))
        x = _rand_i8(jax.random.key(2), (4, 64))
        before = mod(x)
        mod, _ = apply_surgery(mod, rank=2, sigma=0.5, integer_es=True)
        mod.zg_slot.enabled = False
        assert bool(jnp.all(before == mod(x)))

    def test_packed_helper_matches_call(self):
        mod = TernaryLinearLUT(256, 128, use_bias=True, rngs=nnx.Rngs(0))
        x = _rand_i8(jax.random.key(3), (4, 256))
        # On any backend, packed_ternary_lut_fused falls back to the reference
        # when no Triton GPU is present, so it always equals the module call.
        assert bool(jnp.all(packed_ternary_lut_fused(mod, x) == mod(x)))


@GPU_ONLY
@pytest.mark.parametrize(
    "m,k,n",
    [(64, 128, 128), (256, 256, 256), (130, 256, 256), (32, 64, 64)],
)
def test_fused_matches_reference(m, k, n):
    key = jax.random.PRNGKey(m * 7 + k * 13 + n)
    kx, kw, kt = jax.random.split(key, 3)
    x = _rand_i8(kx, (m, k))
    w = _ternary(kw, (k, n))
    table = _rand_i8(kt, (256,))
    packed = pack_ternary(w)
    got = fused_ternary_lut(x, packed, table, mode="fused", allow_fallback=False)
    want = ternary_lut_reference(x, packed, table)
    assert got.dtype == jnp.int8
    assert bool(jnp.all(got == want))


@GPU_ONLY
def test_fused_with_bias():
    k, n = 256, 256
    kx, kw, kb, kt = jax.random.split(jax.random.PRNGKey(0), 4)
    x = _rand_i8(kx, (128, k))
    w = _ternary(kw, (k, n))
    bias = jax.random.randint(kb, (n,), -(2**20), 2**20, jnp.int32)
    table = _rand_i8(kt, (256,))
    packed = pack_ternary(w)
    got = fused_ternary_lut(x, packed, table, bias=bias, mode="fused", allow_fallback=False)
    want = ternary_lut_reference(x, packed, table, bias=bias)
    assert bool(jnp.all(got == want))


@GPU_ONLY
def test_fused_3d_input():
    k, n = 256, 128
    x = _rand_i8(jax.random.PRNGKey(3), (8, 33, k))  # odd T exercises M padding
    w = _ternary(jax.random.PRNGKey(4), (k, n))
    table = _rand_i8(jax.random.PRNGKey(5), (256,))
    packed = pack_ternary(w)
    got = fused_ternary_lut(x, packed, table, mode="fused", allow_fallback=False)
    want = ternary_lut_reference(x, packed, table)
    assert got.shape == (8, 33, n)
    assert bool(jnp.all(got == want))


class TestMixedModelTraining:
    """zerograd + AdamW on a mixed int8 + ternary model keeps per-leaf ranges."""

    class Mixed(nnx.Module):
        def __init__(self, rngs):
            self.fc = IntLinear(64, 64, rngs=rngs)
            self.t1 = TernaryLinearLUT(64, 64, rngs=rngs)
            self.head = IntLinear(64, 10, act_dtype=jnp.int32, rngs=rngs)

        def __call__(self, x):
            return self.head(self.t1(self.fc(x))).astype(jnp.float32)

    def _make(self):
        model = self.Mixed(nnx.Rngs(0))
        opt = ZeroGrad(
            optax.adamw(1e-2),
            population_size=32,
            rank=2,
            seed=0,
            run_id="mixed-ternary-test",
            integer_es=True,
            sigma_shift=2,
            int_bits=8,
        )
        state = opt.init(model)
        return model, opt, state

    def test_mixed_bin_path_auto_enabled(self):
        _, opt, _ = self._make()
        assert not opt._bin_updates  # adamw is not identity
        assert opt._int_bin_updates  # auto-enabled by ternary-marked leaves
        assert opt._int_bits_by_path[("t1", "linear", "kernel")] == 2

    def test_step_keeps_ternary_and_int8_ranges(self):
        model, opt, state = self._make()

        def loss_fn(m, batch):
            x, y = batch
            logits = m(x)
            return jnp.mean(
                optax.softmax_cross_entropy_with_integer_labels(logits, y)
            ), None

        x = _rand_i8(jax.random.key(0), (64, 64))
        y = jax.random.randint(jax.random.key(1), (64,), 0, 10, jnp.int32)
        for _ in range(3):
            model, state, _ = opt.step(state, model, (x, y), loss_fn)
        p = params_pure_dict(model)
        t1k = np.asarray(p["t1"]["linear"]["kernel"])
        fck = np.asarray(p["fc"]["kernel"])
        assert set(np.unique(t1k).tolist()).issubset({-1, 0, 1})
        assert int(fck.min()) >= -128 and int(fck.max()) <= 127
