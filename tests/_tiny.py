"""Shared tiny model + helpers for ZeroGrad integration tests.

Keep fault_tolerant's larger TinyMLP / CE loss local — it differs on purpose.
Optimizer-validation Tiny (kernel-as-output) also stays local.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from zerograd import ZeroGrad

Array = jax.Array


class Tiny(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.l = nnx.Linear(4, 2, rngs=rngs)

    def __call__(self, x):
        return self.l(x)


def build_model(key: Array) -> Tiny:
    return Tiny(nnx.Rngs(key))


def make_model(seed: int = 0) -> Tiny:
    return Tiny(nnx.Rngs(seed))


def mse_loss(model, batch):
    return jnp.sum(model(batch) ** 2), None


def make_opt(pop: int = 8, **kw) -> ZeroGrad:
    defaults = dict(
        transform=optax.adamw(0.01),
        population_size=pop,
        rank=2,
        sigma=0.1,
        seed=42,
        run_id="test",
    )
    defaults.update(kw)
    # Support legacy manifest= kw by ignoring it (NNX auto-manifest).
    defaults.pop("manifest", None)
    return ZeroGrad(**defaults)


def batch() -> Array:
    return jnp.ones((3, 4))
