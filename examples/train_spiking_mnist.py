"""Train a purely spiking MLP on MNIST with ZeroGrad — native SNN modules.

Everything is spiking and gradient-free: pixels are rate-coded into binary
spike trains, ``Spiking`` LIF layers (hard threshold, subtractive reset,
ES-trainable decay/threshold/weights) emit 0/1 throughout, and classification
uses readout spike counts as logits.  ZeroGrad's factor-only ES handles the
non-differentiable dynamics directly — no surrogate gradients.

This is the same recipe that reaches ~83% test accuracy at 5000 gens
(pop 32, rank 8, sigma 0.1, T=16, hidden 256, ~0.1 s/gen on CPU)::

    uv run --with cpu python train_spiking_mnist.py --steps 5000

Recipe highlights (each one is load-bearing for ES on SNNs):

- ``pin_norm=True`` (default): frozen effective weight norms — uniform
  weight growth (the saturation/silence trap) is structurally impossible.
- log1p-count CE: scale-invariant, so uniform rate inflation is neutral and
  only input-dependent reorganization descends.
- Two-sided homeostatic rate targets with weight 100: silence is an ES
  attractor and must be strictly worse than informative firing.
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from zerograd import (
    Spiking,
    ZeroGrad,
    encode_rate,
    spike_count_logits,
)

from _data import load_mnist


class SpikingMLP(nnx.Module):
    def __init__(self, in_dim: int, hidden: int, classes: int, rngs: nnx.Rngs):
        self.h = Spiking(in_dim, hidden, rngs=rngs)
        self.readout = Spiking(hidden, classes, rngs=rngs)

    def __call__(self, spikes: jax.Array) -> jax.Array:
        return self.readout(self.h(spikes))


def main() -> None:
    p = argparse.ArgumentParser(description="Spiking MNIST with ZeroGrad ES")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--T", type=int, default=16)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--rate-target", type=float, default=0.15)
    p.add_argument("--rate-w", type=float, default=100.0)
    p.add_argument("--pop", type=int, default=32)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    x_train, y_train, x_test, y_test = load_mnist()
    print(f"  train {x_train.shape}, test {x_test.shape}")

    model = SpikingMLP(784, args.hidden, 10, rngs=nnx.Rngs(args.seed))
    opt = ZeroGrad(
        optax.adamw(args.lr),
        population_size=args.pop, rank=args.rank, sigma=args.sigma,
        seed=args.seed, run_id="spiking-mnist",
    )
    state = opt.init(model)

    def loss_fn(m, batch, rng=None):
        xb, yb = batch
        spikes = encode_rate(xb, args.T)
        h = m.h(spikes)
        counts = spike_count_logits(m.readout(h))
        ce = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(
            jnp.log1p(counts), yb))
        return (ce
                + args.rate_w * (jnp.mean(h) - args.rate_target) ** 2
                + args.rate_w * (jnp.mean(counts) / args.T - args.rate_target) ** 2,
                None)

    print(f"\nTraining spiking MNIST: T={args.T}, hidden={args.hidden}, "
          f"pop={args.pop}, rank={args.rank}, sigma={args.sigma}\n")
    n_train = x_train.shape[0]
    for step in range(args.steps):
        key = jax.random.fold_in(jax.random.key(args.seed), step)
        idx = jax.random.randint(key, (args.batch,), 0, n_train)
        batch = (x_train[idx], y_train[idx])

        t0 = time.time()
        model, state, metrics = opt.step(state, model, batch, loss_fn)
        dt = time.time() - t0

        if step % 100 == 0 or step == args.steps - 1:
            acc = 0.0
            for i in range(0, x_test.shape[0], 2000):
                counts = spike_count_logits(model(encode_rate(x_test[i:i+2000], args.T)))
                acc += float(jnp.sum(jnp.argmax(counts, -1) == y_test[i:i+2000]))
            acc /= x_test.shape[0]
            print(f"gen {metrics.generation:4d}  mean_loss={metrics.mean_loss:.4f}  "
                  f"min_loss={metrics.min_loss:.4f}  test_acc={acc:.1%}  ({dt:.2f}s/gen)")

    acc = 0.0
    for i in range(0, x_test.shape[0], 2000):
        counts = spike_count_logits(model(encode_rate(x_test[i:i+2000], args.T)))
        acc += float(jnp.sum(jnp.argmax(counts, -1) == y_test[i:i+2000]))
    print(f"\nFinal test accuracy: {acc/x_test.shape[0]:.1%} (chance = 10%)")


if __name__ == "__main__":
    main()