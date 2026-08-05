"""Tests for Eggroll Appendix H integer ES (int8 factors, antithetical, bin updates)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from zerograd import (
    IntLinear,
    ZeroGrad,
    ZgIntLinear,
    apply_bin_updates,
    bin_update_threshold,
    shape_antithetical_loss,
)
import optax
from zerograd._eggroll_h import int8_from_normal, update_alpha_schedule
from zerograd._integer import float_to_int
from zerograd._nnx import params_pure_dict


class TinyInt(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.l1 = IntLinear(2, 8, rngs=rngs)
        self.l2 = IntLinear(8, 1, rngs=rngs)

    def __call__(self, x):
        from zerograd._integer import int_relu

        return self.l2(int_relu(self.l1(x)))


class TestEggrollH:
    def test_int8_factors_are_int8(self):
        a = int8_from_normal(jax.random.key(0), (16, 4))
        assert a.dtype == jnp.int8
        assert int(a.min()) >= -128
        assert int(a.max()) <= 127

    def test_antithetical_shaping(self):
        losses = jnp.asarray([1.0, 3.0, 4.0, 0.5], dtype=jnp.float32)
        # pairs (0,2) and (1,3): sign(4-1)=+1, sign(0.5-3)=-1
        f = shape_antithetical_loss(losses)
        assert f.shape == (2,)
        assert int(f[0]) == 1
        assert int(f[1]) == -1

    def test_bin_threshold_positive(self):
        t = bin_update_threshold(0.5, num_directions=32)
        assert t > 0

    def test_vector_threshold_below_matrix(self):
        """VECTOR evidence is one factor; MATRIX/TABLE use A@B (two factors)."""
        from zerograd import ParameterLayout

        n = 128
        alpha = 0.12
        v = bin_update_threshold(alpha, n, layout=ParameterLayout.VECTOR)
        m = bin_update_threshold(alpha, n, layout=ParameterLayout.MATRIX)
        t = bin_update_threshold(alpha, n, layout=ParameterLayout.TABLE)
        assert v > 0
        assert m == t
        # 16× smaller scale (second factor of 16) → vector bar is far lower.
        assert v * 10 < m

    def test_bin_update_moves_int8(self):
        w = jnp.zeros((4, 4), dtype=jnp.int8)
        e = jnp.array(
            [[100, 0, -100, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            dtype=jnp.int32,
        )
        out = apply_bin_updates(w, e, threshold=50)
        assert out.dtype == jnp.int8
        assert int(out[0, 0]) == 1
        assert int(out[0, 2]) == -1
        assert int(out[0, 1]) == 0

    def test_threshold_tree_scales_lut_vs_kernel(self):
        from zerograd import IntLinear, IntLUT, ParameterLayout, threshold_tree_for_manifest
        from zerograd._nnx import params_pure_dict

        class M(nnx.Module):
            def __init__(self, rngs):
                self.l = IntLinear(4, 4, rngs=rngs)
                self.act = IntLUT(init="identity", rngs=rngs)

            def __call__(self, x):
                return self.l(self.act(x))

        model = M(nnx.Rngs(0))
        opt = ZeroGrad(
            population_size=32,
            rank=2,
            seed=0,
            run_id="thr-tree",
            integer_es=True,
        )
        opt.init(model)
        params = params_pure_dict(model)
        thr = threshold_tree_for_manifest(
            params, opt.manifest, alpha=0.12, num_directions=16
        )
        assert thr["act"]["table"] == bin_update_threshold(
            0.12, 16, layout=ParameterLayout.VECTOR
        )
        assert thr["l"]["kernel"] == bin_update_threshold(
            0.12, 16, layout=ParameterLayout.MATRIX
        )
        assert thr["act"]["table"] < thr["l"]["kernel"]

    def test_alpha_schedule_decays(self):
        a0 = update_alpha_schedule(0, base=1.0, decay=0.015)
        a10 = update_alpha_schedule(10, base=1.0, decay=0.015)
        assert a0 == 1.0
        assert a10 < a0

    def test_integer_es_step_keeps_int_weights(self):
        model = TinyInt(nnx.Rngs(0))
        opt = ZeroGrad(
            population_size=8,
            rank=1,
            seed=0,
            run_id="h-int",
            integer_es=True,
            sigma_shift=4,
        )
        state = opt.init(model)
        assert isinstance(model.l1, ZgIntLinear)
        assert model.zg_slot.integer_es

        def loss_fn(model, batch):
            return jnp.mean(model(batch).astype(jnp.float32) ** 2), None

        x = float_to_int(jnp.ones((4, 2)))
        model, state, metrics = opt.step(state, model, x, loss_fn)
        after = params_pure_dict(model)
        assert all(jnp.issubdtype(v.dtype, jnp.integer) for v in jax.tree.leaves(after))
        assert state.generation == 1
        assert metrics.population_size == 8

    def test_integer_es_loss_can_drop(self):
        model = TinyInt(nnx.Rngs(1))
        opt = ZeroGrad(
            population_size=16,
            rank=1,
            seed=1,
            run_id="h-drop",
            integer_es=True,
            sigma_shift=3,
            update_alpha=1.0,
            alpha_decay=0.0,
        )
        state = opt.init(model)
        x = float_to_int(jnp.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]))
        y = jnp.array([[0.0], [1.0], [1.0], [0.0]])

        def loss_fn(model, batch):
            bx, by = batch
            pred = model(bx).astype(jnp.float32)
            return jnp.mean((pred - by) ** 2), None

        losses = []
        for _ in range(30):
            model, state, m = opt.step(state, model, (x, y), loss_fn)
            losses.append(float(m.mean_loss))
        assert min(losses) < losses[0]

    def test_integer_es_adam_snaps_and_drops(self):
        """Smoke: Adam+snap keeps int weights and weakly reduces MSE.

        Stronger learning checks live in ``test_integer_es_adam_learn.py``.
        """
        model = TinyInt(nnx.Rngs(2))
        opt = ZeroGrad(
            optax.adam(learning_rate=5.0),
            population_size=16,
            rank=1,
            seed=2,
            run_id="h-adam",
            integer_es=True,
            sigma_shift=3,
            int_bits=8,
        )
        state = opt.init(model)
        assert state.opt_state is not None
        assert not opt._bin_updates
        x = float_to_int(jnp.ones((4, 2)))

        def loss_fn(model, batch):
            return jnp.mean(model(batch).astype(jnp.float32) ** 2), None

        losses = []
        for _ in range(20):
            model, state, m = opt.step(state, model, x, loss_fn)
            losses.append(float(m.mean_loss))
        after = params_pure_dict(model)
        assert all(jnp.issubdtype(v.dtype, jnp.integer) for v in jax.tree.leaves(after))
        assert min(losses) < losses[0]
