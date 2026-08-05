"""Train a 4-bit quantized MLP to solve XOR — no Straight-Through Estimator.

This is the killer use case for zero-gradient optimization.  In standard
Quantization-Aware Training (QAT) with backprop, ``jnp.round()`` has zero
gradient almost everywhere, so practitioners hack around it with the
Straight-Through Estimator (STE): ``x_q = x + stop_gradient(round(x) - x)``.

ZeroGrad never differentiates through the forward pass.  The pseudo-gradient
comes from the ES perturbation of *parameters*, not from differentiating the
loss w.r.t. the weights.  So ``jnp.round()`` — and any other non-differentiable
or zero-gradient operation — can be used freely in the forward pass with no
STE, no Gumbel-softmax, no surrogate loss.

The model: same 2→16→1 XOR MLP, but every linear output is quantized to 4-bit
precision (16 levels, signed: -8..+7) using a *learned* scale parameter that is
itself optimized by ZeroGrad.

    uv run python examples/train_qat_xor.py

Compare with ``train_xor.py`` (full-precision) to see that quantization costs
almost nothing in convergence quality here, while being impossible to train
with vanilla backprop + round() without STE.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from zerograd import ZeroGrad

# ── 4-bit quantization ───────────────────────────────────────────────────────

NUM_BITS = 4
QMIN = -(2 ** (NUM_BITS - 1))   # -8
QMAX = 2 ** (NUM_BITS - 1) - 1  # +7


def quantize_4bit(x: jax.Array, scale: jax.Array) -> jax.Array:
    """Quantize ``x`` to signed 4-bit integers, then dequantize.

    ``jnp.round`` has zero gradient almost everywhere — that's the whole point.
    With backprop this function is unusable without STE.  With ZeroGrad it's
    just another operation in the forward pass.
    """
    x_scaled = jnp.clip(x / scale, QMIN, QMAX)
    x_rounded = jnp.round(x_scaled)  # ← zero gradient here — and it's fine
    return x_rounded * scale


# ── Data ────────────────────────────────────────────────────────────────────
X = jnp.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
Y = jnp.array([0, 1, 1, 0])

HIDDEN = 16


class QatXorMLP(nnx.Module):
    """2→16→1 MLP with learned per-channel 4-bit activation scales."""

    def __init__(self, rngs: nnx.Rngs):
        self.l1 = nnx.Linear(2, HIDDEN, rngs=rngs)
        self.l2 = nnx.Linear(HIDDEN, 1, rngs=rngs)
        self.s1 = nnx.Param(jnp.ones((HIDDEN,)) * 0.1)
        self.s2 = nnx.Param(jnp.ones((1,)) * 0.1)

    def __call__(self, x):
        h = self.l1(x)
        h = quantize_4bit(h, self.s1[...])
        h = jnp.maximum(h, 0.0)
        logits = self.l2(h)
        return quantize_4bit(logits, self.s2[...])


def model_loss(model, batch):
    x, y = batch
    logits = jnp.squeeze(model(x), -1)
    loss = optax.sigmoid_binary_cross_entropy(logits, y.astype(jnp.float32))
    return jnp.mean(loss), None


model = QatXorMLP(nnx.Rngs(0))
optimizer = ZeroGrad(
    optax.adamw(learning_rate=1e-2, weight_decay=0.0),
    population_size=32,
    rank=4,
    sigma=0.1,
    seed=0,
    run_id="qat-xor-demo",
)
state = optimizer.init(model)

NUM_STEPS = 500
batch = (X, Y)

for step in range(NUM_STEPS):
    model, state, metrics = optimizer.step(state, model, batch, model_loss)
    if step % 50 == 0 or step == NUM_STEPS - 1:
        logits = jnp.squeeze(model(X), -1)
        preds = (jax.nn.sigmoid(logits) > 0.5).astype(jnp.int32)
        acc = jnp.mean(preds == Y)
        print(
            f"gen {metrics.generation:3d}  "
            f"mean_loss={metrics.mean_loss:.4f}  "
            f"min_loss={metrics.min_loss:.4f}  "
            f"accuracy={float(acc):.1%}"
        )

print("\nFinal predictions:")
logits = jnp.squeeze(model(X), -1)
probs = jax.nn.sigmoid(logits)
for i in range(4):
    print(
        f"  input={list(X[i])}  target={int(Y[i])}  "
        f"prob={float(probs[i]):.3f}  pred={int(probs[i] > 0.5)}"
    )

print(f"\nLearned scales: layer1={list(float(v) for v in model.s1[...])}")
print(f"                layer2={list(float(v) for v in model.s2[...])}")
