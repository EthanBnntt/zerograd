"""Tests for IntLUT nonlinearity, ZgConv surgery, and LUT learning."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from zerograd import (
    IntConv,
    IntLinear,
    IntLinearLUT,
    IntLUT,
    ParameterLayout,
    ZeroGrad,
    ZgConv,
    ZgIntConv,
    ZgIntLinear,
    ZgIntLUT,
    apply_surgery,
    int_avg_pool2d,
)
from zerograd._candidate import perturbed_int_vector
from zerograd._nnx import params_pure_dict


class TinyIntCnn(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.conv = IntConv(1, 4, kernel_size=3, padding="SAME", rngs=rngs)
        self.act = IntLUT(init="identity", explore_shift=0, rngs=rngs)
        self.head = IntLinear(4, 3, act_dtype=jnp.int32, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        # x: int8 NHWC
        h = self.act(self.conv(x))
        h = int_avg_pool2d(h, (h.shape[1], h.shape[2]), (h.shape[1], h.shape[2]))
        h = jnp.reshape(h, (h.shape[0], -1))
        return self.head(h).astype(jnp.float32)


class TestIntLinearLUT:
    def test_forward_is_linear_then_lut(self):
        block = IntLinearLUT(4, 4, use_bias=False, rngs=nnx.Rngs(0))
        x = jnp.ones((2, 4), dtype=jnp.int8)
        y = block(x)
        assert y.dtype == jnp.int8
        assert y.shape == (2, 4)
        # Identity LUT ⇒ same as bare IntLinear.
        assert jnp.array_equal(y, block.lut(block.linear(x)))

    def test_surgery_wraps_children(self):
        class M(nnx.Module):
            def __init__(self, rngs):
                self.block = IntLinearLUT(4, 4, rngs=rngs)

            def __call__(self, x):
                return self.block(x)

        model = M(nnx.Rngs(0))
        model, manifest = apply_surgery(model, rank=2, sigma=0.1, integer_es=True)
        assert isinstance(model.block.linear, ZgIntLinear)
        assert isinstance(model.block.lut, ZgIntLUT)
        paths = {e.path: e.layout for e in manifest.entries}
        assert paths[("block", "linear", "kernel")] is ParameterLayout.MATRIX
        assert paths[("block", "lut", "table")] is ParameterLayout.VECTOR


class TestIntLUT:
    def test_identity_is_linear_map(self):
        lut = IntLUT(init="identity")
        table = lut.table[...]
        assert table.dtype == jnp.int8
        assert table.shape == (256,)
        # -128 → -128, 0 → 0, 127 → 127
        assert int(table[0]) == -128
        assert int(table[128]) == 0
        assert int(table[255]) == 127
        x = jnp.array([-128, -1, 0, 1, 127], dtype=jnp.int8)
        assert jnp.array_equal(lut(x), x)

    def test_relu_init(self):
        lut = IntLUT(init="relu")
        x = jnp.array([-5, 0, 7], dtype=jnp.int8)
        assert jnp.array_equal(lut(x), jnp.array([0, 0, 7], dtype=jnp.int8))

    def test_surgery_wraps_as_zg_int_lut(self):
        class M(nnx.Module):
            def __init__(self, rngs):
                self.act = IntLUT(rngs=rngs)
                self.l = IntLinear(4, 4, rngs=rngs)

            def __call__(self, x):
                return self.l(self.act(x))

        model = M(nnx.Rngs(0))
        model, manifest = apply_surgery(model, rank=2, sigma=0.1, integer_es=True)
        assert isinstance(model.act, ZgIntLUT)
        assert isinstance(model.l, ZgIntLinear)
        paths = {e.path: e.layout for e in manifest.entries}
        assert paths[("act", "table")] is ParameterLayout.VECTOR

    def test_vector_perturbation_is_unbiased(self):
        """Arithmetic ``>>`` biased negatives to -1; trunc-div must not."""
        v = jnp.arange(-128, 128, dtype=jnp.int8)
        # Large sigma_shift → most deltas 0; survivors must not be all-negative.
        p = perturbed_int_vector(v, jax.random.key(0), sigma_shift=4)
        delta = p.astype(jnp.int32) - v.astype(jnp.int32)
        assert int(jnp.max(jnp.abs(delta))) == 0  # |noise| < 256 → 0 after // 256

        p2 = perturbed_int_vector(v, jax.random.key(1), sigma_shift=0)
        d2 = p2.astype(jnp.int32) - v.astype(jnp.int32)
        assert int(jnp.sum(d2 > 0)) > 0
        assert int(jnp.sum(d2 < 0)) > 0

    def test_lut_learns_under_adam_snap(self):
        model = TinyIntCnn(nnx.Rngs(0))
        opt = ZeroGrad(
            optax.adamw(1.0),
            population_size=16,
            rank=2,
            seed=0,
            run_id="lut-learn-adam",
            integer_es=True,
            sigma_shift=2,
            int_bits=8,
            candidate_chunk_size=1,
        )
        state = opt.init(model)
        before = params_pure_dict(model)["act"]["table"].copy()
        assert jnp.array_equal(before, jnp.arange(-128, 128, dtype=jnp.int8))

        def loss_fn(model, batch):
            return jnp.mean(model(batch) ** 2), None

        x = jnp.ones((8, 6, 6, 1), dtype=jnp.int8)
        moved = False
        for _ in range(15):
            model, state, _ = opt.step(state, model, x, loss_fn)
            after = params_pure_dict(model)["act"]["table"]
            if int(jnp.sum(after != before)) > 0:
                moved = True
                break
        assert moved, "IntLUT table should leave identity under AdamW→snap"

    def test_lut_learns_under_layout_scaled_bins(self):
        """Vector-scaled threshold must let LUT bins fire (matrix bar alone never does)."""
        model = TinyIntCnn(nnx.Rngs(0))
        opt = ZeroGrad(
            population_size=64,
            rank=4,
            seed=0,
            run_id="lut-learn-bins",
            integer_es=True,
            sigma_shift=2,
            update_alpha=0.2,
            candidate_chunk_size=1,
        )
        state = opt.init(model)
        assert opt._bin_updates
        before = params_pure_dict(model)["act"]["table"].copy()

        def loss_fn(model, batch):
            return jnp.mean(model(batch) ** 2), None

        x = jnp.ones((8, 6, 6, 1), dtype=jnp.int8)
        moved = False
        for _ in range(25):
            model, state, _ = opt.step(state, model, x, loss_fn)
            after = params_pure_dict(model)["act"]["table"]
            if int(jnp.sum(after != before)) > 0:
                moved = True
                break
        assert moved, "IntLUT should receive ±1 bins with vector-scaled threshold"


class TestIntConv:
    def test_surgery_wraps_int_conv(self):
        class M(nnx.Module):
            def __init__(self, rngs):
                self.conv = IntConv(3, 8, kernel_size=3, rngs=rngs)

            def __call__(self, x):
                return self.conv(x)

        model = M(nnx.Rngs(0))
        assert model.conv.kernel[...].shape == (3 * 3 * 3, 8)
        model, manifest = apply_surgery(model, rank=2, sigma=0.05, integer_es=True)
        assert isinstance(model.conv, ZgIntConv)
        assert any(e.path == ("conv", "kernel") for e in manifest.entries)

    def test_forward_int8(self):
        m = IntConv(1, 4, kernel_size=3, rngs=nnx.Rngs(0))
        x = jnp.ones((2, 8, 8, 1), dtype=jnp.int8)
        y = m(x)
        assert y.dtype == jnp.int8
        assert y.shape == (2, 8, 8, 4)

    def test_pure_int_step_keeps_dtypes(self):
        model = TinyIntCnn(nnx.Rngs(0))
        opt = ZeroGrad(
            optax.adamw(1.0),
            population_size=8,
            rank=2,
            seed=0,
            run_id="int-cnn",
            integer_es=True,
            sigma_shift=2,
            int_bits=8,
            candidate_chunk_size=1,
        )
        state = opt.init(model)
        assert isinstance(model.conv, ZgIntConv)
        assert isinstance(model.act, ZgIntLUT)

        def loss_fn(model, batch):
            return jnp.mean(model(batch) ** 2), None

        x = jnp.ones((4, 6, 6, 1), dtype=jnp.int8)
        model, state, _ = opt.step(state, model, x, loss_fn)
        params = params_pure_dict(model)
        assert all(jnp.issubdtype(v.dtype, jnp.integer) for v in jax.tree.leaves(params))
        assert state.generation == 1


class TestZgConv:
    def test_surgery_flattens_float_conv_kernel(self):
        class M(nnx.Module):
            def __init__(self, rngs):
                self.conv = nnx.Conv(3, 8, kernel_size=(3, 3), padding="SAME", rngs=rngs)

            def __call__(self, x):
                return self.conv(x)

        model = M(nnx.Rngs(0))
        assert model.conv.kernel[...].shape == (3, 3, 3, 8)
        model, manifest = apply_surgery(model, rank=2, sigma=0.05)
        assert isinstance(model.conv, ZgConv)
        assert model.conv.kernel[...].shape == (3 * 3 * 3, 8)
        assert any(e.path == ("conv", "kernel") for e in manifest.entries)
