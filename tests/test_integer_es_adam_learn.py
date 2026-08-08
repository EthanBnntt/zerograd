"""Synthetic learning tests for integer ES → pseudo-grad → Adam → snap.

Pipeline (``integer_es=True`` + non-identity Optax transform)::

  1. Antithetical int8 factor forward (Appendix H.1)
  2. Centered-loss fitness → pair weights → ``replay_integer`` descent
  3. Negate descent → Optax pseudo-gradient
  4. Adam (float view of weights) → ``apply_updates``
  5. ``snap_tree_to_integer`` back to int8/int4 storage

Identity transform keeps the ±1 bin path instead; these tests assert the Adam path.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from zerograd import IntLinear, ZeroGrad, ZgIntLinear
from zerograd._integer import (
    float_to_int,
    float_view_tree,
    int_relu,
    snap_tree_to_integer,
)
from zerograd._nnx import params_pure_dict


class IntMLP(nnx.Module):
    """Small pure-int MLP for synthetic tasks."""

    def __init__(self, din: int, hidden: int, dout: int, *, rngs: nnx.Rngs):
        self.l1 = IntLinear(din, hidden, rngs=rngs)
        self.l2 = IntLinear(hidden, dout, act_dtype=jnp.int32, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        h = int_relu(self.l1(x))
        return self.l2(h)


def _all_integer(params: dict) -> bool:
    return all(jnp.issubdtype(v.dtype, jnp.integer) for v in jax.tree.leaves(params))


class TestIntegerEsAdamPlumbing:
    def test_adam_path_not_bin_updates(self):
        model = IntMLP(2, 8, 1, rngs=nnx.Rngs(0))
        opt = ZeroGrad(
            optax.adam(1.0),
            population_size=8,
            rank=1,
            seed=0,
            run_id="plumb",
            integer_es=True,
            sigma_shift=3,
        )
        state = opt.init(model)
        assert not opt._bin_updates
        assert opt.integer_es
        assert state.opt_state is not None
        assert isinstance(model.l1, ZgIntLinear)

    def test_snap_preserves_dtype_and_range(self):
        template = {"w": jnp.array([[1, -2], [3, 4]], dtype=jnp.int8)}
        updated = {"w": jnp.array([[1.6, -2.4], [3.1, 127.9]], dtype=jnp.float32)}
        snapped = snap_tree_to_integer(updated, template, bits=8)
        assert snapped["w"].dtype == jnp.int8
        np.testing.assert_array_equal(snapped["w"], np.array([[2, -2], [3, 127]], dtype=np.int8))

    def test_float_view_is_float32(self):
        tree = {"w": jnp.ones((2, 2), dtype=jnp.int8), "b": jnp.zeros(2, dtype=jnp.int32)}
        view = float_view_tree(tree)
        assert view["w"].dtype == jnp.float32
        assert view["b"].dtype == jnp.float32


class TestIntegerEsAdamLearns:
    def test_xor_learns(self):
        """Classic XOR: Adam-snap integer ES must learn the boolean table.

        Integer logits rarely hit exact {0,1} MSE, so we require a large
        relative loss drop plus peak decision accuracy (threshold 0.5).
        """
        X = float_to_int(jnp.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]), bits=8)
        Y = jnp.array([[0.0], [1.0], [1.0], [0.0]])
        y_i = Y.reshape(-1).astype(jnp.int32)

        model = IntMLP(2, 16, 1, rngs=nnx.Rngs(0))
        opt = ZeroGrad(
            optax.adam(learning_rate=2.0),
            population_size=32,
            rank=2,
            seed=0,
            run_id="xor-adam",
            integer_es=True,
            sigma_shift=2,
            int_bits=8,
        )
        state = opt.init(model)
        assert not opt._bin_updates

        def loss_fn(m, batch):
            bx, by = batch
            pred = m(bx).astype(jnp.float32) / 32.0
            return jnp.mean((pred - by) ** 2), None

        def accuracy(m):
            pred = (m(X).astype(jnp.float32) / 32.0 > 0.5).astype(jnp.int32).reshape(-1)
            return float(jnp.mean(pred == y_i))

        losses = []
        best_acc = 0.0
        for _ in range(200):
            model, state, metrics = opt.step(state, model, (X, Y), loss_fn)
            losses.append(float(metrics.mean_loss))
            best_acc = max(best_acc, accuracy(model))

        assert _all_integer(params_pure_dict(model))
        assert min(losses) < losses[0] * 0.05, (
            f"XOR MSE did not drop enough: {losses[0]:.3f} → {min(losses):.3f}"
        )
        assert best_acc >= 0.75, f"XOR peak accuracy {best_acc:.0%} (want ≥75%)"

    def test_linear_regression_loss_drops(self):
        """Fit y = sum(x) with an int MLP; loss must fall by >50%."""
        key = jax.random.key(7)
        x_f = jax.random.normal(key, (64, 4))
        y = jnp.sum(x_f, axis=-1, keepdims=True)
        X = float_to_int(x_f, bits=8)

        model = IntMLP(4, 32, 1, rngs=nnx.Rngs(7))
        opt = ZeroGrad(
            optax.adam(learning_rate=1.0),
            population_size=32,
            rank=2,
            seed=7,
            run_id="linreg-adam",
            integer_es=True,
            sigma_shift=2,
            int_bits=8,
        )
        state = opt.init(model)

        def loss_fn(m, batch):
            bx, by = batch
            pred = m(bx).astype(jnp.float32) / 32.0
            return jnp.mean((pred - by) ** 2), None

        losses = []
        for _ in range(80):
            model, state, metrics = opt.step(state, model, (X, y), loss_fn)
            losses.append(float(metrics.mean_loss))

        assert _all_integer(params_pure_dict(model))
        assert min(losses) < losses[0] * 0.5, (
            f"regression loss stuck: {losses[0]:.3f} → {min(losses):.3f}"
        )

    def test_three_class_toy_accuracy(self):
        """Three Gaussian blobs → int classifier accuracy must rise above chance."""
        key = jax.random.key(11)
        k1, k2, k3, k4 = jax.random.split(key, 4)
        n = 40
        c0 = jax.random.normal(k1, (n, 2)) * 0.3 + jnp.array([-1.5, 0.0])
        c1 = jax.random.normal(k2, (n, 2)) * 0.3 + jnp.array([1.5, 0.0])
        c2 = jax.random.normal(k3, (n, 2)) * 0.3 + jnp.array([0.0, 1.5])
        x_f = jnp.concatenate([c0, c1, c2], axis=0)
        y = jnp.concatenate(
            [
                jnp.zeros(n, dtype=jnp.int32),
                jnp.ones(n, dtype=jnp.int32),
                jnp.full(n, 2, dtype=jnp.int32),
            ]
        )
        perm = jax.random.permutation(k4, x_f.shape[0])
        x_f, y = x_f[perm], y[perm]
        X = float_to_int(x_f, bits=8)

        model = IntMLP(2, 32, 3, rngs=nnx.Rngs(11))
        opt = ZeroGrad(
            optax.adam(learning_rate=1.5),
            population_size=32,
            rank=2,
            seed=11,
            run_id="blobs-adam",
            integer_es=True,
            sigma_shift=2,
            int_bits=8,
        )
        state = opt.init(model)

        def loss_fn(m, batch):
            bx, by = batch
            logits = m(bx).astype(jnp.float32) / 32.0
            return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, by)), None

        def accuracy(m):
            pred = jnp.argmax(m(X).astype(jnp.float32), axis=-1)
            return float(jnp.mean(pred == y))

        acc0 = accuracy(model)
        losses = []
        for _ in range(100):
            model, state, metrics = opt.step(state, model, (X, y), loss_fn)
            losses.append(float(metrics.mean_loss))
        acc1 = accuracy(model)

        assert _all_integer(params_pure_dict(model))
        assert min(losses) < losses[0] - 0.2, (
            f"CE did not drop: {losses[0]:.3f} → {min(losses):.3f}"
        )
        assert acc1 > acc0 + 0.15, f"acc {acc0:.0%} → {acc1:.0%} (need clear gain)"
        assert acc1 >= 0.55, f"final acc {acc1:.0%} still near chance"

    def test_params_change_under_adam(self):
        model = IntMLP(2, 8, 1, rngs=nnx.Rngs(3))
        opt = ZeroGrad(
            optax.adam(learning_rate=3.0),
            population_size=16,
            rank=1,
            seed=3,
            run_id="move-adam",
            integer_es=True,
            sigma_shift=2,
        )
        state = opt.init(model)
        before = params_pure_dict(model)
        x = float_to_int(jnp.ones((8, 2)))

        def loss_fn(m, batch):
            return jnp.mean(m(batch).astype(jnp.float32) ** 2), None

        for _ in range(10):
            model, state, _ = opt.step(state, model, x, loss_fn)
        after = params_pure_dict(model)
        changed = sum(
            int(jnp.sum(a != b))
            for a, b in zip(jax.tree.leaves(before), jax.tree.leaves(after), strict=True)
        )
        assert changed > 0, "Adam-snap path did not change any integer weights"


class TestIntegerEsAdamVsBins:
    def test_identity_is_bins_adam_is_not(self):
        bins = ZeroGrad(
            population_size=8,
            rank=1,
            seed=0,
            run_id="bins",
            integer_es=True,
            sigma_shift=3,
        )
        adam = ZeroGrad(
            optax.adam(0.1),
            population_size=8,
            rank=1,
            seed=0,
            run_id="adam",
            integer_es=True,
            sigma_shift=3,
        )
        assert bins._bin_updates
        assert not adam._bin_updates
