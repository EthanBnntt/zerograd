"""Shared training loop for the MNIST / CIFAR-10 2-layer MLP examples.

``train_mnist.py`` and `train_cifar10.py`` are thin wrappers around
:func:`run` that only differ in dataset, dimensions, and a couple of
hyperparameters (see issue #33-style duplication cleanup). Keeping one
parameterized training loop here means fixes (checkpointing, early
stopping, logging) land in both scripts at once.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import optax

from zerograd import Manifest, ManifestEntry, ParameterLayout, ZeroGrad

from _checkpoint import EarlyStopping, load_checkpoint, save_checkpoint


@dataclass(frozen=True)
class VisionMlpSpec:
    """Dataset-specific knobs for the shared 2-layer MLP training loop."""

    name: str
    load_data: Callable[[], tuple]
    input_dim: int
    hidden: int
    default_steps: int
    default_batch: int
    lr: float
    sigma: float
    run_id: str
    manifest_prefix: str
    num_classes: int = 10


def build_params(key: jax.Array, spec: VisionMlpSpec) -> dict:
    """He-initialized ``input_dim -> hidden -> num_classes`` MLP parameters."""
    k1, k2 = jax.random.fold_in(key, 1), jax.random.fold_in(key, 2)
    w1 = jax.random.normal(k1, (spec.input_dim, spec.hidden)) * jnp.sqrt(2.0 / spec.input_dim)
    b1 = jnp.zeros((spec.hidden,))
    w2 = jax.random.normal(k2, (spec.hidden, spec.num_classes)) * jnp.sqrt(2.0 / spec.hidden)
    b2 = jnp.zeros((spec.num_classes,))
    return {
        "layer1": {"weight": w1, "bias": b1},
        "layer2": {"weight": w2, "bias": b2},
    }


def build_manifest(spec: VisionMlpSpec) -> Manifest:
    prefix = spec.manifest_prefix
    return Manifest(
        version=1,
        entries=(
            ManifestEntry(("layer1", "weight"), ParameterLayout.MATRIX, f"{prefix}_w1"),
            ManifestEntry(("layer1", "bias"), ParameterLayout.VECTOR, f"{prefix}_b1"),
            ManifestEntry(("layer2", "weight"), ParameterLayout.MATRIX, f"{prefix}_w2"),
            ManifestEntry(("layer2", "bias"), ParameterLayout.VECTOR, f"{prefix}_b2"),
        ),
    )


def model_loss(params, candidate, batch, rng):
    del rng
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


def build_arg_parser(spec: VisionMlpSpec) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Train {spec.name} MLP with ZeroGrad")
    parser.add_argument("--steps", type=int, default=spec.default_steps)
    parser.add_argument("--batch", type=int, default=spec.default_batch)
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
    return parser


def run(spec: VisionMlpSpec, args: argparse.Namespace) -> None:
    """Run the shared ZeroGrad MLP training loop for ``spec``."""
    print(f"Loading {spec.name} ...")
    x_train, y_train, x_test, y_test = spec.load_data()
    x_train = jnp.array(x_train)
    y_train = jnp.array(y_train)
    x_test = jnp.array(x_test)
    y_test = jnp.array(y_test)
    print(f"  train: {x_train.shape}, test: {x_test.shape}")

    key = jax.random.key(args.seed)
    params = build_params(key, spec)

    optimizer = ZeroGrad(
        build_manifest(spec),
        optax.adamw(learning_rate=spec.lr, weight_decay=0.0),
        population_size=32,
        rank=8,
        sigma=spec.sigma,
        seed=args.seed,
        run_id=spec.run_id,
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

    step = start_step
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


def main(spec: VisionMlpSpec) -> None:
    args = build_arg_parser(spec).parse_args()
    run(spec, args)
