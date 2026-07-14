"""Train an MLP classifier on MNIST using ZeroGrad evolutionary optimization.

Model: 784 → 64 → 10, ReLU hidden, softmax cross-entropy loss.

ZeroGrad evaluates 32 perturbed candidates per step (no backprop), so each
step costs ~32× a forward pass.  On CPU, expect ~2-4 seconds per step.
Expected accuracy after 200 steps: ~85-92%.  This won't match gradient-based
training (98%+) — ES trades sample efficiency for the ability to optimize
through non-differentiable operations and arbitrary objectives.

    uv run python examples/train_mnist.py [--steps N] [--batch N]

Data is downloaded on first run to ~/.cache/zerograd/ (~12 MB).
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import optax

from zerograd import Manifest, ManifestEntry, ParameterLayout, ZeroGrad

from _checkpoint import EarlyStopping, load_checkpoint, save_checkpoint
from _data import load_mnist

# ── Config ───────────────────────────────────────────────────────────────────
DEFAULT_STEPS = 200
DEFAULT_BATCH = 128
INPUT_DIM = 784
HIDDEN = 64
NUM_CLASSES = 10
LR = 1e-2  # ES pseudo-gradients need a higher LR than typical gradient training


def build_params(key: jax.Array) -> dict:
    """He-initialized 784→64→10 MLP parameters."""
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
            ManifestEntry(("layer1", "weight"), ParameterLayout.MATRIX, "mnist_w1"),
            ManifestEntry(("layer1", "bias"),   ParameterLayout.VECTOR, "mnist_b1"),
            ManifestEntry(("layer2", "weight"), ParameterLayout.MATRIX, "mnist_w2"),
            ManifestEntry(("layer2", "bias"),   ParameterLayout.VECTOR, "mnist_b2"),
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
    parser = argparse.ArgumentParser(description="Train MNIST MLP with ZeroGrad")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to write periodic checkpoints (params + state).")
    parser.add_argument("--checkpoint-interval", type=int, default=20,
                        help="Save a checkpoint every N steps.")
    parser.add_argument("--early-stopping", action="store_true",
                        help="Stop early when the loss plateaus.")
    parser.add_argument("--patience", type=int, default=50,
                        help="Early-stopping patience (steps without improvement).")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from a checkpoint file.")
    args = parser.parse_args()

    print("Loading MNIST ...")
    x_train, y_train, x_test, y_test = load_mnist()
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
        sigma=0.1,
        seed=args.seed,
        run_id="mnist-demo",
    )
    start_step = 0
    if args.resume:
        ck = load_checkpoint(args.resume)
        params = ck["params"]
        state = ck["state"]
        start_step = ck["step"] + 1
        print(f"Resumed from {args.resume} at step {start_step}")
    else:
        state = optimizer.init(params)

    num_train = x_train.shape[0]
    es = EarlyStopping(patience=args.patience, mode="min") if args.early_stopping else None
    print(f"\nTraining: {args.steps} steps, pop=32, batch={args.batch}\n")

    for step in range(start_step, args.steps):
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

        if args.checkpoint and (step + 1) % args.checkpoint_interval == 0:
            save_checkpoint(args.checkpoint, step, params, state)
            print(f"  checkpoint saved: {args.checkpoint}")
        if es is not None and es(float(metrics.mean_loss)):
            print(f"Early stopping at step {step}: loss plateaued for {args.patience} steps.")
            break

    print(f"\nFinal test accuracy: {float(evaluate(params, x_test, y_test)):.1%}")
    if args.checkpoint:
        save_checkpoint(args.checkpoint, step, params, state)


if __name__ == "__main__":
    main()
