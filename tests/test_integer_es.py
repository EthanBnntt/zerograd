"""Tests for integer ES: int8 factors, antithetical shaping, and bin updates."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from zerograd import (
    IntLinear,
    IntLUT,
    ParameterLayout,
    ZeroGrad,
    ZgIntLinear,
    apply_bin_updates,
    apply_bin_updates_jit,
    bin_update_threshold,
    shape_antithetical_loss,
    threshold_tree_for_manifest,
)
from zerograd._bin_updates import update_alpha_schedule
from zerograd._factors import int8_from_normal
from zerograd._integer import float_to_int, int_relu
from zerograd._nnx import params_pure_dict


class TinyInt(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.l1 = IntLinear(2, 8, rngs=rngs)
        self.l2 = IntLinear(8, 1, rngs=rngs)

    def __call__(self, x):
        return self.l2(int_relu(self.l1(x)))


class TestIntegerEs:
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
        n = 128
        alpha = 0.12
        v = bin_update_threshold(alpha, n, layout=ParameterLayout.VECTOR)
        m = bin_update_threshold(alpha, n, layout=ParameterLayout.MATRIX)
        t = bin_update_threshold(alpha, n, layout=ParameterLayout.TABLE)
        assert v > 0
        assert m == t
        # 16× smaller scale (second factor of 16) → vector bar is far lower.
        assert v * 10 < m

    def test_matrix_threshold_scales_with_rank(self):
        """MATRIX/TABLE null std ∝ √rank; bar must track or flip rate inflates."""
        n, alpha = 256, 0.12
        t1 = bin_update_threshold(alpha, n, layout=ParameterLayout.MATRIX, rank=1)
        t4 = bin_update_threshold(alpha, n, layout=ParameterLayout.MATRIX, rank=4)
        # √4 = 2 → threshold doubles (integer floor).
        assert t4 == int(t1 * 2) or abs(t4 / t1 - 2.0) < 0.02
        # Vectors have no rank axis in evidence.
        v1 = bin_update_threshold(alpha, n, layout=ParameterLayout.VECTOR, rank=1)
        v4 = bin_update_threshold(alpha, n, layout=ParameterLayout.VECTOR, rank=4)
        assert v1 == v4

    def test_null_flip_rate_stable_across_rank(self):
        """Under random ±1 weights, realized flip fraction ≈ alpha for rank 1..8."""
        from zerograd._keys import step_key
        from zerograd._replay import replay_integer

        half = 256
        alpha = 0.12
        rates = []
        for rank in (1, 2, 4, 8):
            model = TinyInt(nnx.Rngs(4))
            opt = ZeroGrad(
                population_size=512,
                rank=rank,
                seed=4,
                run_id=f"null-r{rank}",
                integer_es=True,
                sigma_shift=2,
                update_alpha=alpha,
                alpha_decay=0.0,
            )
            opt.init(model)
            params = params_pure_dict(model)
            shaped = jax.random.choice(
                jax.random.key(99 + rank),
                jnp.array([-1, 1], dtype=jnp.int32),
                (half,),
            )
            base_key = step_key(4, f"null-r{rank}", 0, opt.manifest.version)
            pair_ids = jnp.arange(half, dtype=jnp.int32)
            evidence = replay_integer(
                params, opt.manifest, base_key, pair_ids, shaped, rank
            )
            thr = threshold_tree_for_manifest(
                params,
                opt.manifest,
                alpha=alpha,
                num_directions=half,
                rank=rank,
            )
            new_p = apply_bin_updates(params, evidence, thr)
            total = changed = 0
            for a, b in zip(jax.tree.leaves(params), jax.tree.leaves(new_p), strict=True):
                if not jnp.issubdtype(a.dtype, jnp.integer):
                    continue
                total += int(a.size)
                changed += int(jnp.sum(a != b))
            rates.append(changed / max(total, 1))
        # All ranks should land near alpha (binomial noise on ~65 ints).
        for rate in rates:
            assert abs(rate - alpha) < 0.12, rates
        assert max(rates) - min(rates) < 0.15, rates

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

    def test_apply_bin_updates_jit_matches_eager(self):
        params = {"w": jnp.array([0, 10, 20], dtype=jnp.int8)}
        evidence = {"w": jnp.array([100, -100, 5], dtype=jnp.int32)}
        thresholds = {"w": 50}
        eager = apply_bin_updates(params, evidence, thresholds)
        jitted = apply_bin_updates_jit(params, evidence, thresholds)
        np.testing.assert_array_equal(np.asarray(eager["w"]), np.asarray(jitted["w"]))

    def test_threshold_tree_scales_lut_vs_kernel(self):
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
            params, opt.manifest, alpha=0.12, num_directions=16, rank=2
        )
        assert thr["act"]["table"] == bin_update_threshold(
            0.12, 16, layout=ParameterLayout.VECTOR, rank=2
        )
        assert thr["l"]["kernel"] == bin_update_threshold(
            0.12, 16, layout=ParameterLayout.MATRIX, rank=2
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
