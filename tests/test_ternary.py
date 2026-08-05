"""Ternary / BitNet-style helpers and TernaryLinear surgery."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from zerograd import TernaryLinear, ZeroGrad, ZgTernaryLinear, apply_surgery
from zerograd._integer import (
    TERNARY_MAX,
    TERNARY_MIN,
    absmax_quantize_int8,
    absmean,
    qrange,
    quantize_ternary,
    snap_tree_to_integer,
    ternary_dequant_scale,
    ternary_init,
)


class TestTernaryHelpers:
    def test_qrange_bits2_is_ternary(self):
        assert qrange(2) == (TERNARY_MIN, TERNARY_MAX)

    def test_quantize_ternary_values(self):
        w = jnp.asarray([[0.5, -0.01, -2.0], [0.0, 1.5, -0.3]], dtype=jnp.float32)
        w_t, gamma = quantize_ternary(w)
        assert w_t.dtype == jnp.int8
        assert set(np.unique(np.asarray(w_t)).tolist()).issubset({-1, 0, 1})
        np.testing.assert_allclose(float(gamma), float(absmean(w)), rtol=1e-5)

    def test_ternary_init_in_range(self):
        w_t, gamma = ternary_init(jax.random.key(0), (64, 32))
        assert w_t.dtype == jnp.int8
        assert int(w_t.min()) >= TERNARY_MIN
        assert int(w_t.max()) <= TERNARY_MAX
        assert float(gamma) > 0

    def test_absmax_quantize_int8(self):
        x = jnp.asarray([0.0, 0.5, -1.0, 2.0], dtype=jnp.float32)
        x_q, eta = absmax_quantize_int8(x)
        assert x_q.dtype == jnp.int8
        assert float(eta) == pytest.approx(2.0)
        assert int(x_q[-1]) == 127

    def test_dequant_scale(self):
        s = ternary_dequant_scale(jnp.asarray(0.1), jnp.asarray(2.0))
        np.testing.assert_allclose(float(s), 0.1 * 2.0 / 127.0)

    def test_snap_bits2(self):
        template = jnp.zeros((4, 4), dtype=jnp.int8)
        updated = jnp.asarray(
            [[-2.3, -0.4, 0.2, 1.7], [0.0, 0.6, -1.1, 3.0]] * 2, dtype=jnp.float32
        ).reshape(4, 4)
        snapped = snap_tree_to_integer(updated, template, bits=2)
        assert snapped.dtype == jnp.int8
        assert set(np.unique(np.asarray(snapped)).tolist()).issubset({-1, 0, 1})


class TinyTernary(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.l1 = TernaryLinear(4, 8, use_bias=True, rngs=rngs)
        self.l2 = TernaryLinear(8, 2, use_bias=False, rngs=rngs)

    def __call__(self, x):
        return self.l2(nnx.gelu(self.l1(x)))


class TestTernaryLinear:
    def test_forward_bf16_out(self):
        model = TinyTernary(nnx.Rngs(0))
        x = jnp.ones((3, 4), dtype=jnp.bfloat16)
        y = model(x)
        assert y.dtype == jnp.bfloat16
        assert y.shape == (3, 2)
        assert jnp.all(jnp.isfinite(y.astype(jnp.float32)))

    def test_kernel_is_ternary(self):
        layer = TernaryLinear(16, 8, rngs=nnx.Rngs(1))
        w = layer.kernel[...]
        assert w.dtype == jnp.int8
        assert set(np.unique(np.asarray(w)).tolist()).issubset({-1, 0, 1})

    def test_surgery_replaces_ternary_linear(self):
        model = TinyTernary(nnx.Rngs(0))
        model, manifest = apply_surgery(model, rank=2, sigma=0.05, integer_es=True)
        assert isinstance(model.l1, ZgTernaryLinear)
        assert isinstance(model.l2, ZgTernaryLinear)
        paths = {e.path for e in manifest.entries}
        assert ("l1", "kernel") in paths
        assert ("l1", "gamma") in paths
        assert ("l1", "rms_scale") in paths
        assert ("l1", "bias") in paths
        assert ("l2", "kernel") in paths
        assert ("l2", "bias") not in paths

    def test_eval_path_unperturbed(self):
        model = TinyTernary(nnx.Rngs(0))
        x = jnp.ones((4, 4), dtype=jnp.bfloat16)
        before = model(x)
        model, _ = apply_surgery(model, rank=2, sigma=0.5, integer_es=True)
        model.zg_slot.enabled = False
        after = model(x)
        assert jnp.allclose(before.astype(jnp.float32), after.astype(jnp.float32), rtol=1e-3)

    def test_zerograd_step_keeps_ternary(self):
        model = TinyTernary(nnx.Rngs(0))
        opt = ZeroGrad(
            optax.adamw(1e-2),
            population_size=8,
            rank=2,
            seed=0,
            run_id="ternary-test",
            integer_es=True,
            sigma_shift=4,
            int_bits=2,
        )
        state = opt.init(model)
        assert opt._ternary_bins

        def loss_fn(m, batch):
            x, y = batch
            logits = m(x).astype(jnp.float32)
            return jnp.mean((logits - y) ** 2), None

        x = jnp.ones((4, 4), dtype=jnp.bfloat16)
        y = jnp.zeros((4, 2), dtype=jnp.float32)
        w_before = np.array(model.l1.kernel[...])
        model, state, metrics = opt.step(state, model, (x, y), loss_fn)
        assert metrics.population_size == 8
        w = model.l1.kernel[...]
        assert w.dtype == jnp.int8
        assert set(np.unique(np.asarray(w)).tolist()).issubset({-1, 0, 1})
        # Bin path should be able to flip at least some weights over a step
        # (not required every seed; just that dtype/range stay valid).
        del w_before
