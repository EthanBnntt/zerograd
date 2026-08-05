"""Train a small MLP to regress a sine wave using ZeroGrad evolutionary optimization.

Demonstrates that ZeroGrad can optimize for continuous regression targets —
not just classification — purely from population fitness shaping.  The model
learns to approximate sin(x) over [-π, π] without any gradient computation.

Runs in ~30 seconds on CPU with no external data.

    uv run python examples/train_sine.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from zerograd import ZeroGrad

# ── Data ────────────────────────────────────────────────────────────────────
key = jax.random.key(42)
x_all = jnp.linspace(-jnp.pi, jnp.pi, 200).reshape(-1, 1)
y_all = jnp.sin(x_all) + 0.05 * jax.random.normal(key, x_all.shape)

HIDDEN = 32


class SineMLP(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.l1 = nnx.Linear(1, HIDDEN, rngs=rngs)
        self.l2 = nnx.Linear(HIDDEN, 1, rngs=rngs)

    def __call__(self, x):
        return self.l2(jnp.tanh(self.l1(x)))


def model_loss(model, batch):
    x, y = batch
    return jnp.mean((model(x) - y) ** 2), None


model = SineMLP(nnx.Rngs(0))
optimizer = ZeroGrad(
    optax.adamw(learning_rate=3e-3, weight_decay=0.0),
    population_size=32,
    rank=4,
    sigma=0.1,
    seed=0,
    run_id="sine-demo",
)
state = optimizer.init(model)

NUM_STEPS = 1000
BATCH_SIZE = 64
NUM_SAMPLES = x_all.shape[0]

for step in range(NUM_STEPS):
    idx = jax.random.randint(jax.random.fold_in(key, step), (BATCH_SIZE,), 0, NUM_SAMPLES)
    batch = (x_all[idx], y_all[idx])

    model, state, metrics = optimizer.step(state, model, batch, model_loss)
    if step % 100 == 0 or step == NUM_STEPS - 1:
        full_mse = jnp.mean((model(x_all) - y_all) ** 2)
        print(
            f"gen {metrics.generation:4d}  "
            f"pop_mean={metrics.mean_loss:.4f}  "
            f"pop_min={metrics.min_loss:.4f}  "
            f"full_mse={float(full_mse):.4f}"
        )

print("\nSample predictions (x → target → predicted):")
out = model(x_all)
for i in range(0, 200, 25):
    print(
        f"  x={float(x_all[i, 0]):+.3f}  target={float(y_all[i, 0]):+.3f}  "
        f"pred={float(out[i, 0]):+.3f}"
    )
