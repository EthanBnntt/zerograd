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
from flax import nnx

from zerograd import ZeroGrad

# ── Data ────────────────────────────────────────────────────────────────────
X = jnp.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
Y = jnp.array([0, 1, 1, 0])  # XOR truth table

# ── Model: 2 → 16 → 1 MLP ────────────────────────────────────────────────────
HIDDEN = 16


class XorMLP(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.l1 = nnx.Linear(2, HIDDEN, rngs=rngs)
        self.l2 = nnx.Linear(HIDDEN, 1, rngs=rngs)

    def __call__(self, x):
        return self.l2(nnx.relu(self.l1(x)))


def model_loss(model, batch):
    x, y = batch
    logits = jnp.squeeze(model(x), -1)
    loss = optax.sigmoid_binary_cross_entropy(logits, y.astype(jnp.float32))
    return jnp.mean(loss), None


model = XorMLP(nnx.Rngs(0))
optimizer = ZeroGrad(
    optax.adamw(learning_rate=1e-2, weight_decay=0.0),
    population_size=32,
    rank=4,
    sigma=0.1,
    seed=0,
    run_id="xor-demo",
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
