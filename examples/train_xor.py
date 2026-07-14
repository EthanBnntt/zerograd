"""Train a tiny MLP to solve XOR using ZeroGrad evolutionary optimization.

The XOR problem is the classic non-linearly-separable benchmark: no single
linear decision boundary can separate the four points, so the network needs a
hidden layer.  This script demonstrates that ZeroGrad — which never computes a
gradient through the model — can still learn non-linear structure purely from
population fitness shaping.

Runs in a few seconds on CPU with no external data.

    uv run python examples/train_xor.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

from zerograd import Manifest, ManifestEntry, ParameterLayout, ZeroGrad

# ── Data ────────────────────────────────────────────────────────────────────
# All four XOR rows, full-batch every step.
X = jnp.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
Y = jnp.array([0, 1, 1, 0])  # XOR truth table

# ── Model ────────────────────────────────────────────────────────────────────
# 2 → 16 → 1 MLP with ReLU hidden activation and sigmoid output.
HIDDEN = 16
key = jax.random.key(0)
w1 = jax.random.normal(jax.random.fold_in(key, 1), (2, HIDDEN)) * 0.5
b1 = jnp.zeros((HIDDEN,))
w2 = jax.random.normal(jax.random.fold_in(key, 2), (HIDDEN, 1)) * 0.5
b2 = jnp.zeros((1,))

params = {
    "layer1": {"weight": w1, "bias": b1},
    "layer2": {"weight": w2, "bias": b2},
}

manifest = Manifest(
    version=1,
    entries=(
        ManifestEntry(("layer1", "weight"), ParameterLayout.MATRIX, "xor_w1"),
        ManifestEntry(("layer1", "bias"), ParameterLayout.VECTOR, "xor_b1"),
        ManifestEntry(("layer2", "weight"), ParameterLayout.MATRIX, "xor_w2"),
        ManifestEntry(("layer2", "bias"), ParameterLayout.VECTOR, "xor_b2"),
    ),
)


def model_loss(params, candidate, batch, rng):
    x, y = batch
    h = candidate.linear(params, ("layer1", "weight"), x)
    h = h + candidate.vector(params, ("layer1", "bias"))
    h = jnp.maximum(h, 0.0)  # ReLU
    logits = candidate.linear(params, ("layer2", "weight"), h)
    logits = logits + candidate.vector(params, ("layer2", "bias"))
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
    run_id="xor-demo",
)

state = optimizer.init(params)

# ── Training loop ────────────────────────────────────────────────────────────
NUM_STEPS = 500
batch = (X, Y)

for step in range(NUM_STEPS):
    params, state, metrics = optimizer.step(state, params, batch, model_loss)
    if step % 50 == 0 or step == NUM_STEPS - 1:
        # Evaluate current params (unperturbed) for accuracy
        h = jnp.maximum(params["layer1"]["weight"].T @ X.T + params["layer1"]["bias"][:, None], 0.0)
        logits = (params["layer2"]["weight"].T @ h).squeeze(0) + params["layer2"]["bias"][0]
        preds = (jax.nn.sigmoid(logits) > 0.5).astype(jnp.int32)
        acc = jnp.mean(preds == Y)
        print(
            f"gen {metrics.generation:3d}  "
            f"mean_loss={metrics.mean_loss:.4f}  "
            f"min_loss={metrics.min_loss:.4f}  "
            f"accuracy={float(acc):.1%}"
        )

print("\nFinal predictions:")
h = jnp.maximum(params["layer1"]["weight"].T @ X.T + params["layer1"]["bias"][:, None], 0.0)
logits = (params["layer2"]["weight"].T @ h).squeeze(0) + params["layer2"]["bias"][0]
probs = jax.nn.sigmoid(logits)
for i in range(4):
    print(f"  input={list(X[i])}  target={int(Y[i])}  prob={float(probs[i]):.3f}  pred={int(probs[i] > 0.5)}")
