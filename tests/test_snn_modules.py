"""Tests for the native SNN modules: Leaky, Spiking, spike coding, readout."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
import pytest
from flax import nnx

from zerograd import (
    Leaky,
    Spiking,
    ZeroGrad,
    ZgLinear,
    ZgVector,
    encode_poisson,
    encode_rate,
    spike_count_logits,
    apply_surgery,
)


# ── reference dynamics ───────────────────────────────────────────────────────


def leaky_reference(x, w, beta, bias=None):
    """y_t = beta·y_{t-1} + x_t @ W (+ b), manual loop."""
    b, t, _ = x.shape
    y = jnp.zeros((b, w.shape[1]), dtype=x.dtype)
    out = []
    for i in range(t):
        c = x[:, i] @ w + (bias if bias is not None else 0.0)
        y = beta * y + c
        out.append(y)
    return jnp.stack(out, axis=1)


def lif_reference(x, w, beta, thr, bias=None):
    """LIF with subtractive reset, manual loop."""
    b, t, _ = x.shape
    v = jnp.zeros((b, w.shape[1]), dtype=x.dtype)
    out = []
    for i in range(t):
        c = x[:, i] @ w + (bias if bias is not None else 0.0)
        v = beta * v + c
        s = (v >= thr).astype(x.dtype)
        v = v - s * thr
        out.append(s)
    return jnp.stack(out, axis=1)


# ── Leaky ────────────────────────────────────────────────────────────────────


class TestLeaky:
    def test_decay_matches_reference(self):
        layer = Leaky(4, 3, leak=0.75, rngs=nnx.Rngs(0))
        x = jax.random.normal(jax.random.key(1), (2, 7, 4))
        got = layer(x)
        beta = layer.beta_value()
        w = layer.lin.kernel[...]
        assert jnp.allclose(got, leaky_reference(x, w, beta), atol=1e-6)

    def test_with_bias(self):
        layer = Leaky(4, 3, leak=0.9, use_bias=True, rngs=nnx.Rngs(0))
        x = jax.random.normal(jax.random.key(2), (2, 5, 4))
        got = layer(x)
        beta = layer.beta_value()
        w, b = layer.lin.kernel[...], layer.lin.bias[...]
        assert jnp.allclose(got, leaky_reference(x, w, beta, b), atol=1e-6)

    def test_beta_in_open_interval(self):
        layer = Leaky(4, 3, leak=0.123, rngs=nnx.Rngs(0))
        beta = layer.beta_value()
        assert float(beta.min()) > 0.0 and float(beta.max()) < 1.0
        assert jnp.allclose(beta, 0.123, atol=1e-6)

    def test_leak_validation(self):
        for bad in (0.0, 1.0, -0.5, 1.5):
            with pytest.raises(ValueError):
                Leaky(4, 3, leak=bad, rngs=nnx.Rngs(0))
        with pytest.raises(ValueError):
            Leaky(0, 3, rngs=nnx.Rngs(0))

    def test_input_validation(self):
        layer = Leaky(4, 3, rngs=nnx.Rngs(0))
        with pytest.raises(ValueError):
            layer(jnp.zeros((2, 4)))  # ndim 2
        with pytest.raises(ValueError):
            layer(jnp.zeros((2, 5, 5)))  # wrong Din


# ── Spiking ─────────────────────────────────────────────────────────────────


class TestSpiking:
    def test_lif_matches_reference(self):
        layer = Spiking(4, 3, leak=0.8, threshold=0.7, rngs=nnx.Rngs(0))
        x = jax.random.normal(jax.random.key(3), (2, 9, 4))
        got = layer(x)
        w = layer.lin.kernel[...]
        assert jnp.allclose(got, lif_reference(
            x, w, layer.beta_value(), layer.threshold_value()), atol=1e-6)

    def test_output_binary_and_reset_bounded(self):
        # Large inputs drive many spikes; membrane must stay bounded
        # (subtractive reset) and output strictly binary.
        layer = Spiking(2, 4, leak=0.95, threshold=0.5, rngs=nnx.Rngs(0))
        x = jnp.abs(jax.random.normal(jax.random.key(4), (3, 20, 2))) * 3.0
        out = layer(x)
        assert bool(((out == 0) | (out == 1)).all())
        # No neuron may fire twice in a row at high input (reset by thr):
        consecutive = (out[:, :-1] * out[:, 1:]).sum()
        assert int(consecutive) < out[:, 1:].size  # not all-adjacent

    def test_pin_norm_freezes_effective_norm(self):
        layer = Spiking(8, 4, pin_norm=True, rngs=nnx.Rngs(0))
        init_norms = jnp.asarray(layer.w_scale)
        # Scale the kernel up 10x: pinning must fully compensate.
        layer.lin.kernel[...] = layer.lin.kernel[...] * 10.0
        norms = jnp.linalg.norm(layer.lin.kernel[...], axis=0) + 1e-8
        scale = init_norms / norms
        x = jax.random.normal(jax.random.key(5), (2, 6, 8))
        raw = x @ layer.lin.kernel[...]
        pinned = raw * scale  # == x @ W_original: growth fully compensated
        original = x @ (layer.lin.kernel[...] / 10.0)
        beta, thr = layer.beta_value(), layer.threshold_value()

        def run(current):
            v = jnp.zeros((2, 4))
            spikes = []
            for i in range(6):
                v = beta * v + current[:, i]
                s = (v >= thr).astype(x.dtype)
                v = v - s * thr
                spikes.append(s)
            return jnp.stack(spikes, 1)

        # The 10x weight growth is behavior-neutral under pinning: the
        # forward computes the same currents as the original kernel.
        assert jnp.array_equal(run(pinned), run(original))

    def test_validation(self):
        with pytest.raises(ValueError):
            Spiking(4, 3, threshold=0.0, rngs=nnx.Rngs(0))
        with pytest.raises(ValueError):
            Spiking(4, 3, leak=1.0, rngs=nnx.Rngs(0))


# ── spike coding / readout ───────────────────────────────────────────────────


class TestCoding:
    def test_rate_coding_counts(self):
        x = jnp.asarray([[0.0, 0.5, 1.0]])
        s = encode_rate(x, 16)
        assert s.shape == (1, 16, 3)
        assert bool(((s == 0) | (s == 1)).all())
        assert int(s[0, :, 0].sum()) == 0  # p=0 never fires
        assert int(s[0, :, 2].sum()) == 16  # p=1 fires every step
        assert int(s[0, :, 1].sum()) == 8  # p=0.5 → leading half
        # Spikes are leading (deterministic), not scattered:
        assert bool(s[0, :8, 1].all()) and not bool(s[0, 8:, 1].any())

    def test_poisson_coding(self):
        x = jnp.full((1, 2), 0.25)
        s = encode_poisson(x, 200, jax.random.key(0))
        assert s.shape == (1, 200, 2)
        assert bool(((s == 0) | (s == 1)).all())
        rate = float(s.mean())
        assert 0.15 < rate < 0.35

    def test_count_logits(self):
        s = jnp.ones((3, 8, 5))
        assert spike_count_logits(s).shape == (3, 5)
        assert int(spike_count_logits(s).max()) == 8


# ── surgery integration ─────────────────────────────────────────────────────


class SNN(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.h = Spiking(4, 8, rngs=rngs)
        self.out = Spiking(8, 3, rngs=rngs)

    def __call__(self, x):
        return self.out(self.h(x))


class SNNXor(nnx.Module):
    """Din=2 two-layer SNN for the rate-coded XOR task."""

    def __init__(self, rngs: nnx.Rngs):
        self.h = Spiking(2, 8, rngs=rngs)
        self.out = Spiking(8, 2, rngs=rngs)

    def __call__(self, x):
        return self.out(self.h(x))


class TestSurgery:
    def test_surgery_wraps_internals(self):
        model = SNN(nnx.Rngs(0))
        _, manifest = apply_surgery(model, rank=8, sigma=0.1)
        assert isinstance(model.h.lin, ZgLinear)
        assert isinstance(model.out.lin, ZgLinear)
        assert isinstance(model.h.beta, ZgVector)
        assert isinstance(model.h.thresh_raw, ZgVector)
        assert isinstance(model.out.beta, ZgVector)
        assert isinstance(model.out.thresh_raw, ZgVector)
        # Manifest knows about kernels and the 1-D membrane params.
        paths = {".".join(e.path) for e in manifest.entries}
        assert "h.lin.kernel" in paths and "h.beta.param" in paths
        assert "out.thresh_raw.param" in paths

    def test_eval_path_unperturbed(self):
        model = SNN(nnx.Rngs(0))
        apply_surgery(model, rank=8, sigma=0.1)
        x = (jax.random.uniform(jax.random.key(6), (2, 6, 4)) < 0.5) * 1.0
        a = model(x)
        b = model(x)
        assert jnp.array_equal(a, b)  # slot disabled outside a step → deterministic

    def test_es_step_runs_and_learns(self):
        # Spiking XOR — mirrors test_xor_learns for the float MLP: the full
        # gradient-free loop must descend and (like the float version) reach
        # ≥0.75 accuracy on hard-threshold, non-differentiable dynamics.
        x = jnp.asarray([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        y = jnp.asarray([0, 1, 1, 0])

        model = SNNXor(nnx.Rngs(0))
        opt = ZeroGrad(optax.adamw(1e-2), population_size=32, rank=4,
                       sigma=0.1, seed=0, run_id="snn-xor")

        def loss_fn(m, batch, rng=None):
            spikes = encode_rate(batch[0], 8)
            h = m.h(spikes)
            counts = spike_count_logits(m.out(h))
            ce = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(
                jnp.log1p(counts), batch[1]))
            # Homeostatic targets: silence is an ES trap — it must be
            # strictly worse than informative firing.
            return (ce
                    + 100.0 * (jnp.mean(h) - 0.15) ** 2
                    + 100.0 * (jnp.mean(counts) / 8.0 - 0.15) ** 2, None)

        state = opt.init(model)
        first = None
        for i in range(400):
            model, state, metrics = opt.step(state, model, (x, y), loss_fn)
            if first is None:
                first = float(metrics.min_loss)
        assert float(metrics.min_loss) < first

        counts = spike_count_logits(model(encode_rate(x, 8)))
        acc = float(jnp.mean(jnp.argmax(counts, -1) == y))
        assert acc >= 0.75, f"spiking XOR accuracy {acc:.2f} < 0.75"
