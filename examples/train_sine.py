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

from zerograd import Manifest, ManifestEntry, ParameterLayout, ZeroGrad

# ── Data ────────────────────────────────────────────────────────────────────
# 200 samples of sin(x) over [-π, π], plus small Gaussian noise.
key = jax.random.key(42)
x_all = jnp.linspace(-jnp.pi, jnp.pi, 200).reshape(-1, 1)
y_all = jnp.sin(x_all) + 0.05 * jax.random.normal(key, x_all.shape)

# ── Model ────────────────────────────────────────────────────────────────────
# 1 → 32 → 1 MLP with tanh hidden activation (natural periodicity helps).
HIDDEN = 32
w1 = jax.random.normal(jax.random.fold_in(key, 1), (1, HIDDEN)) * 0.5
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
        ManifestEntry(("layer1", "weight"), ParameterLayout.MATRIX, "sine_w1"),
        ManifestEntry(("layer1", "bias"), ParameterLayout.VECTOR, "sine_b1"),
        ManifestEntry(("layer2", "weight"), ParameterLayout.MATRIX, "sine_w2"),
        ManifestEntry(("layer2", "bias"), ParameterLayout.VECTOR, "sine_b2"),
    ),
)


def model_loss(params, candidate, batch, rng):
    x, y = batch
    h = candidate.linear(params, ("layer1", "weight"), x)
    h = h + candidate.vector(params, ("layer1", "bias"))
    h = jnp.tanh(h)
    out = candidate.linear(params, ("layer2", "weight"), h)
    out = out + candidate.vector(params, ("layer2", "bias"))
    return jnp.mean((out - y) ** 2), None  # MSE


# ── Optimizer ────────────────────────────────────────────────────────────────
optimizer = ZeroGrad(
    manifest,
    optax.adamw(learning_rate=3e-3, weight_decay=0.0),
    population_size=32,
    rank=4,
    sigma=0.1,
    seed=0,
    run_id="sine-demo",
)

state = optimizer.init(params)

# ── Training loop ────────────────────────────────────────────────────────────
NUM_STEPS = 1000
BATCH_SIZE = 64
NUM_SAMPLES = x_all.shape[0]

for step in range(NUM_STEPS):
    # Random mini-batch each step
    idx = jax.random.randint(jax.random.fold_in(key, step), (BATCH_SIZE,), 0, NUM_SAMPLES)
    batch = (x_all[idx], y_all[idx])

    params, state, metrics = optimizer.step(state, params, batch, model_loss)
    if step % 100 == 0 or step == NUM_STEPS - 1:
        # Evaluate on full dataset
        h = jnp.tanh(x_all @ params["layer1"]["weight"] + params["layer1"]["bias"])
        out = h @ params["layer2"]["weight"] + params["layer2"]["bias"]
        full_mse = jnp.mean((out - y_all) ** 2)
        print(
            f"gen {metrics.generation:4d}  "
            f"pop_mean={metrics.mean_loss:.4f}  "
            f"pop_min={metrics.min_loss:.4f}  "
            f"full_mse={float(full_mse):.4f}"
        )

# ── Final sample ─────────────────────────────────────────────────────────────
print("\nSample predictions (x → target → predicted):")
h = jnp.tanh(x_all @ params["layer1"]["weight"] + params["layer1"]["bias"])
out = h @ params["layer2"]["weight"] + params["layer2"]["bias"]
for i in range(0, 200, 25):
    print(f"  x={float(x_all[i, 0]):+.3f}  target={float(y_all[i, 0]):+.3f}  pred={float(out[i, 0]):+.3f}")
