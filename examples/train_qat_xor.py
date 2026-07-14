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
itself optimized by ZeroGrad.  The scale is a manifest VECTOR entry, so it
receives its own ES perturbations and is replayed into its own pseudo-gradient.

    uv run python examples/train_qat_xor.py

Compare with ``train_xor.py`` (full-precision) to see that quantization costs
almost nothing in convergence quality here, while being impossible to train
with vanilla backprop + round() without STE.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

from zerograd import Manifest, ManifestEntry, ParameterLayout, ZeroGrad

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

# ── Model ────────────────────────────────────────────────────────────────────
# 2 → 16 → 1 MLP with ReLU, 4-bit quantization after each linear layer.
# Each layer has a weight (MATRIX), bias (VECTOR), and quantization scale (VECTOR).
HIDDEN = 16
key = jax.random.key(0)
w1 = jax.random.normal(jax.random.fold_in(key, 1), (2, HIDDEN)) * 0.5
b1 = jnp.zeros((HIDDEN,))
s1 = jnp.ones((HIDDEN,)) * 0.1   # learned per-channel quantization scale
w2 = jax.random.normal(jax.random.fold_in(key, 2), (HIDDEN, 1)) * 0.5
b2 = jnp.zeros((1,))
s2 = jnp.ones((1,)) * 0.1

params = {
    "layer1": {"weight": w1, "bias": b1, "scale": s1},
    "layer2": {"weight": w2, "bias": b2, "scale": s2},
}

manifest = Manifest(
    version=1,
    entries=(
        ManifestEntry(("layer1", "weight"), ParameterLayout.MATRIX, "qat_w1"),
        ManifestEntry(("layer1", "bias"),   ParameterLayout.VECTOR, "qat_b1"),
        ManifestEntry(("layer1", "scale"),  ParameterLayout.VECTOR, "qat_s1"),
        ManifestEntry(("layer2", "weight"), ParameterLayout.MATRIX, "qat_w2"),
        ManifestEntry(("layer2", "bias"),   ParameterLayout.VECTOR, "qat_b2"),
        ManifestEntry(("layer2", "scale"),  ParameterLayout.VECTOR, "qat_s2"),
    ),
)


def model_loss(params, candidate, batch, rng):
    x, y = batch

    # Layer 1: linear → bias → quantize → ReLU
    h = candidate.linear(params, ("layer1", "weight"), x)
    h = h + candidate.vector(params, ("layer1", "bias"))
    s1 = candidate.vector(params, ("layer1", "scale"))
    h = quantize_4bit(h, s1)           # ← zero-gradient op, no STE needed
    h = jnp.maximum(h, 0.0)            # ReLU

    # Layer 2: linear → bias → quantize
    logits = candidate.linear(params, ("layer2", "weight"), h)
    logits = logits + candidate.vector(params, ("layer2", "bias"))
    s2 = candidate.vector(params, ("layer2", "scale"))
    logits = quantize_4bit(logits, s2)  # ← zero-gradient op, no STE needed
    logits = jnp.squeeze(logits, axis=-1)

    loss = optax.sigmoid_binary_cross_entropy(logits, y.astype(jnp.float32))
    return jnp.mean(loss), None


# ── Optimizer ────────────────────────────────────────────────────────────────
optimizer = ZeroGrad(
    manifest,
    optax.adamw(learning_rate=1e-2, weight_decay=0.0),
    population_size=32,
    rank=4,
    sigma=0.1,
    seed=0,
    run_id="qat-xor-demo",
)

state = optimizer.init(params)

# ── Training loop ────────────────────────────────────────────────────────────
NUM_STEPS = 500
batch = (X, Y)

for step in range(NUM_STEPS):
    params, state, metrics = optimizer.step(state, params, batch, model_loss)
    if step % 50 == 0 or step == NUM_STEPS - 1:
        # Evaluate current params (unperturbed) for accuracy
        h = params["layer1"]["weight"].T @ X.T + params["layer1"]["bias"][:, None]
        h = quantize_4bit(h, params["layer1"]["scale"][:, None])
        h = jnp.maximum(h, 0.0)
        logits = (params["layer2"]["weight"].T @ h).squeeze(0) + params["layer2"]["bias"]
        logits = quantize_4bit(logits, params["layer2"]["scale"])
        preds = (jax.nn.sigmoid(logits) > 0.5).astype(jnp.int32)
        acc = jnp.mean(preds == Y)
        print(
            f"gen {metrics.generation:3d}  "
            f"mean_loss={metrics.mean_loss:.4f}  "
            f"min_loss={metrics.min_loss:.4f}  "
            f"accuracy={float(acc):.1%}"
        )

# ── Final report ─────────────────────────────────────────────────────────────
print("\n4-bit QAT XOR — no Straight-Through Estimator used.")
print(f"Quantization: {NUM_BITS}-bit signed ({QMIN}..{QMAX}) with learned scale.\n")
print("Predictions:")
h = params["layer1"]["weight"].T @ X.T + params["layer1"]["bias"][:, None]
h = quantize_4bit(h, params["layer1"]["scale"][:, None])
h = jnp.maximum(h, 0.0)
logits = (params["layer2"]["weight"].T @ h).squeeze(0) + params["layer2"]["bias"]
logits = quantize_4bit(logits, params["layer2"]["scale"])
probs = jax.nn.sigmoid(logits)
for i in range(4):
    print(f"  input={list(float(v) for v in X[i])}  target={int(Y[i])}  prob={float(probs[i]):.3f}  pred={int(probs[i] > 0.5)}")

print(f"\nLearned scales: layer1={list(float(v) for v in params['layer1']['scale'])}")
print(f"                layer2={list(float(v) for v in params['layer2']['scale'])}")
