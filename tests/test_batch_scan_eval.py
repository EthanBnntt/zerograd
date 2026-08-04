"""Full-population vmap evaluation (no scan)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from zerograd import ZeroGrad


class Tiny(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.l = nnx.Linear(4, 3, rngs=rngs)

    def __call__(self, x):
        return self.l(x)


def test_nnx_full_vmap_eval_finite():
    model = Tiny(nnx.Rngs(0))
    opt = ZeroGrad(
        optax.sgd(0.01),
        population_size=8,
        rank=1,
        sigma=0.05,
        seed=0,
        run_id="full-vmap",
    )
    opt.init(model)
    x = jax.random.normal(jax.random.key(1), (16, 4))
    y = jax.random.randint(jax.random.key(2), (16,), 0, 3)

    def loss_fn(m, b):
        bx, by = b
        return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(m(bx), by)), None

    losses = opt.evaluate_shard(
        model, 0, loss_fn, (x, y), jnp.arange(8, dtype=jnp.int32), rng=jax.random.key(3)
    )
    assert losses.shape == (8,)
    assert jnp.all(jnp.isfinite(losses))
