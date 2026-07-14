"""Shared XOR model definition for the XOR-based example scripts.

Each training script keeps its own training loop; only the model definition
(``build_params``, ``build_manifest``, ``loss_fn``, ``accuracy``) and the XOR
dataset are shared here, to avoid copy-pasted boilerplate across the
distributed and cluster XOR examples (see issue #30).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from zerograd import Manifest, ManifestEntry, ParameterLayout

# ── Model: 2→16→1 MLP on XOR ─────────────────────────────────────────────────
INPUT_DIM = 2
HIDDEN_DIM = 16
OUTPUT_DIM = 1

# ── XOR data ──────────────────────────────────────────────────────────────────
XOR_X = jnp.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
XOR_Y = jnp.array([[0.0], [1.0], [1.0], [0.0]])


def build_params(key):
    """2→16→1 MLP parameters drawn deterministically from ``key``."""
    k1, k2 = jax.random.split(key)
    return {
        "w1": jax.random.normal(k1, (INPUT_DIM, HIDDEN_DIM)) * 0.5,
        "b1": jnp.zeros((HIDDEN_DIM,)),
        "w2": jax.random.normal(k2, (HIDDEN_DIM, OUTPUT_DIM)) * 0.5,
        "b2": jnp.zeros((OUTPUT_DIM,)),
    }


def build_manifest():
    """Manifest for the 2→16→1 MLP parameter tree."""
    return Manifest(version=1, entries=(
        ManifestEntry(("w1",), ParameterLayout.MATRIX, "w1"),
        ManifestEntry(("b1",), ParameterLayout.VECTOR, "b1"),
        ManifestEntry(("w2",), ParameterLayout.MATRIX, "w2"),
        ManifestEntry(("b2",), ParameterLayout.VECTOR, "b2"),
    ))


def loss_fn(params, candidate, batch, rng):
    """XOR mean-squared-error through a tanh MLP using candidate perturbations."""
    x, y = batch
    h = jax.nn.tanh(candidate.linear(params, ("w1",), x))
    h = h + candidate.vector(params, ("b1",))
    logits = candidate.linear(params, ("w2",), h)
    logits = logits + candidate.vector(params, ("b2",))
    return jnp.mean((logits - y) ** 2), None


def accuracy(params):
    """Fraction of XOR examples classified correctly by ``params``."""
    h = jax.nn.tanh(XOR_X @ params["w1"]) + params["b1"]
    logits = h @ params["w2"] + params["b2"]
    preds = (logits > 0.5).astype(jnp.float32)
    return float(jnp.mean(preds == XOR_Y))
