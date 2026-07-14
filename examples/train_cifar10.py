"""Train an MLP classifier on CIFAR-10 using ZeroGrad evolutionary optimization.

Model: 3072 → 128 → 10, ReLU hidden, softmax cross-entropy loss.

CIFAR-10 is a much harder dataset than MNIST for a small MLP — even gradient-
based training tops out around 50-55% with this architecture.  ZeroGrad's ES
approach will reach a more modest accuracy (~35-45%) because:

  1. Evolutionary strategies are less sample-efficient than backprop.
  2. The low-rank perturbation (rank=8) limits the search space dimensionality.
  3. Each step evaluates 32 candidates, not one gradient.
  4. 3072-dim inputs amplify perturbation noise, requiring a smaller sigma.

The point is not to beat backprop — it's to demonstrate that ZeroGrad can
optimize a real image classifier end-to-end using only fitness evaluation,
which enables training through non-differentiable operations (see train_qat_xor.py).

    uv run python examples/train_cifar10.py [--steps N] [--batch N]

Data is downloaded on first run to ~/.cache/zerograd/ (~170 MB).
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import optax

from zerograd import Manifest, ManifestEntry, ParameterLayout, ZeroGrad

from _data import load_cifar10

# ── Config ───────────────────────────────────────────────────────────────────
DEFAULT_STEPS = 300
DEFAULT_BATCH = 128
INPUT_DIM = 3072  # 32×32×3 flattened
HIDDEN = 128
NUM_CLASSES = 10
LR = 5e-3   # ES pseudo-gradients need a higher LR than typical gradient training
SIGMA = 0.02  # lower than MNIST — 3072-dim inputs amplify perturbation noise


def build_params(key: jax.Array) -> dict:
    """He-initialized 3072→128→10 MLP parameters."""
    w1 = jax.random.normal(jax.random.fold_in(key, 1), (INPUT_DIM, HIDDEN)) * jnp.sqrt(2.0 / INPUT_DIM)
    b1 = jnp.zeros((HIDDEN,))
    w2 = jax.random.normal(jax.random.fold_in(key, 2), (HIDDEN, NUM_CLASSES)) * jnp.sqrt(2.0 / HIDDEN)
    b2 = jnp.zeros((NUM_CLASSES,))
    return {
        "layer1": {"weight": w1, "bias": b1},
        "layer2": {"weight": w2, "bias": b2},
    }


def build_manifest() -> Manifest:
    return Manifest(
        version=1,
        entries=(
            ManifestEntry(("layer1", "weight"), ParameterLayout.MATRIX, "cifar_w1"),
            ManifestEntry(("layer1", "bias"),   ParameterLayout.VECTOR, "cifar_b1"),
            ManifestEntry(("layer2", "weight"), ParameterLayout.MATRIX, "cifar_w2"),
            ManifestEntry(("layer2", "bias"),   ParameterLayout.VECTOR, "cifar_b2"),
        ),
    )


def model_loss(params, candidate, batch, rng):
    x, y = batch
    h = candidate.linear(params, ("layer1", "weight"), x)
    h = h + candidate.vector(params, ("layer1", "bias"))
    h = jnp.maximum(h, 0.0)  # ReLU
    logits = candidate.linear(params, ("layer2", "weight"), h)
    logits = logits + candidate.vector(params, ("layer2", "bias"))
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, y)
    return jnp.mean(loss), None


def evaluate(params, x, y):
    h = jnp.maximum(x @ params["layer1"]["weight"] + params["layer1"]["bias"], 0.0)
    logits = h @ params["layer2"]["weight"] + params["layer2"]["bias"]
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == y)


def main():
    parser = argparse.ArgumentParser(description="Train CIFAR-10 MLP with ZeroGrad")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print("Loading CIFAR-10 ...")
    x_train, y_train, x_test, y_test = load_cifar10()
    x_train = jnp.array(x_train)
    y_train = jnp.array(y_train)
    x_test = jnp.array(x_test)
    y_test = jnp.array(y_test)
    print(f"  train: {x_train.shape}, test: {x_test.shape}")

    key = jax.random.key(args.seed)
    params = build_params(key)
    manifest = build_manifest()

    optimizer = ZeroGrad(
        manifest,
        optax.adamw(learning_rate=LR, weight_decay=0.0),
        population_size=32,
        rank=8,
        sigma=SIGMA,
        seed=args.seed,
        run_id="cifar-demo",
    )
    state = optimizer.init(params)

    num_train = x_train.shape[0]
    print(f"\nTraining: {args.steps} steps, pop=32, batch={args.batch}\n")

    for step in range(args.steps):
        idx = jax.random.randint(jax.random.fold_in(key, step), (args.batch,), 0, num_train)
        batch = (x_train[idx], y_train[idx])

        t0 = time.time()
        params, state, metrics = optimizer.step(state, params, batch, model_loss)
        dt = time.time() - t0

        if step % 20 == 0 or step == args.steps - 1:
            test_acc = evaluate(params, x_test, y_test)
            print(
                f"gen {metrics.generation:3d}  "
                f"mean_loss={metrics.mean_loss:.4f}  "
                f"min_loss={metrics.min_loss:.4f}  "
                f"test_acc={float(test_acc):.1%}  "
                f"({dt:.1f}s/step)"
            )

    print(f"\nFinal test accuracy: {float(evaluate(params, x_test, y_test)):.1%}")


if __name__ == "__main__":
    main()
