"""Shared XOR model definition for the XOR-based example scripts.

Each training script keeps its own training loop; only the model definition
(``build_model``, ``loss_fn``, ``accuracy``) and the XOR dataset are shared
here, to avoid copy-pasted boilerplate across the distributed and cluster XOR
examples.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

# ── Model: 2→16→1 MLP on XOR ─────────────────────────────────────────────────
INPUT_DIM = 2
HIDDEN_DIM = 16
OUTPUT_DIM = 1

# ── XOR data ──────────────────────────────────────────────────────────────────
XOR_X = jnp.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
XOR_Y = jnp.array([[0.0], [1.0], [1.0], [0.0]])


class XorMLP(nnx.Module):
    """2→16→1 tanh MLP for the XOR demos."""

    def __init__(self, rngs: nnx.Rngs):
        self.l1 = nnx.Linear(INPUT_DIM, HIDDEN_DIM, rngs=rngs)
        self.l2 = nnx.Linear(HIDDEN_DIM, OUTPUT_DIM, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.l2(jnp.tanh(self.l1(x)))


def build_model(key: jax.Array) -> XorMLP:
    """Deterministic XOR MLP from a PRNG key (seed-derived cluster builders)."""
    return XorMLP(nnx.Rngs(key))


def loss_fn(model: XorMLP, batch) -> tuple[jax.Array, None]:
    """XOR mean-squared-error through the tanh MLP."""
    x, y = batch
    return jnp.mean((model(x) - y) ** 2), None


def accuracy(model: XorMLP) -> float:
    """Fraction of XOR examples classified correctly by ``model``."""
    logits = model(XOR_X)
    preds = (logits > 0.5).astype(jnp.float32)
    return float(jnp.mean(preds == XOR_Y))
