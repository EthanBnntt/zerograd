"""Shared training loop for the MNIST / CIFAR-10 2-layer MLP examples.

``train_vision_mlp.py`` are thin wrappers around
:func:`run` that only differ in dataset, dimensions, and a couple of
hyperparameters. Keeping one parameterized training loop here means fixes
(checkpointing, early stopping, logging) land in every dataset at once.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax
from _checkpoint import EarlyStopping, load_checkpoint, save_checkpoint
from flax import nnx

from zerograd import ZeroGrad
from zerograd._nnx import params_pure_dict, update_params


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
    num_classes: int = 10


class VisionMLP(nnx.Module):
    """Flattened-image MLP: input_dim → hidden → num_classes."""

    def __init__(self, input_dim: int, hidden: int, num_classes: int, *, rngs: nnx.Rngs):
        self.l1 = nnx.Linear(input_dim, hidden, rngs=rngs)
        self.l2 = nnx.Linear(hidden, num_classes, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.l2(nnx.relu(self.l1(x)))


def model_loss(model: VisionMLP, batch) -> tuple[jax.Array, None]:
    x, y = batch
    logits = model(x)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, y)
    return jnp.mean(loss), None


def evaluate(model: VisionMLP, x, y) -> jax.Array:
    preds = jnp.argmax(model(x), axis=-1)
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

    model = VisionMLP(
        spec.input_dim, spec.hidden, spec.num_classes, rngs=nnx.Rngs(args.seed)
    )

    optimizer = ZeroGrad(
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
        state = optimizer.init(model)
        update_params(model, ck["params"])
        state = ck["state"]
        start_step = ck["step"] + 1
        print(f"Resumed from {args.resume} at step {start_step}")
    else:
        state = optimizer.init(model)

    num_train = x_train.shape[0]
    es = EarlyStopping(patience=args.patience, mode="min") if args.early_stopping else None
    print(f"\nTraining: {args.steps} steps, pop=32, batch={args.batch}\n")

    step = start_step
    for step in range(start_step, args.steps):
        idx = jax.random.randint(jax.random.fold_in(jax.random.key(args.seed), step), (args.batch,), 0, num_train)
        batch = (x_train[idx], y_train[idx])

        t0 = time.time()
        model, state, metrics = optimizer.step(state, model, batch, model_loss)
        dt = time.time() - t0

        if step % 20 == 0 or step == args.steps - 1:
            test_acc = evaluate(model, x_test, y_test)
            print(
                f"gen {metrics.generation:3d}  "
                f"mean_loss={metrics.mean_loss:.4f}  "
                f"min_loss={metrics.min_loss:.4f}  "
                f"test_acc={float(test_acc):.1%}  "
                f"({dt:.1f}s/step)"
            )

        if args.checkpoint and (step + 1) % args.checkpoint_interval == 0:
            save_checkpoint(args.checkpoint, step, params_pure_dict(model), state)
            print(f"  checkpoint saved: {args.checkpoint}")
        if es is not None and es(float(metrics.mean_loss)):
            print(f"Early stopping at step {step}: loss plateaued for {args.patience} steps.")
            break

    print(f"\nFinal test accuracy: {float(evaluate(model, x_test, y_test)):.1%}")
    if args.checkpoint:
        save_checkpoint(args.checkpoint, step, params_pure_dict(model), state)


def main(spec: VisionMlpSpec) -> None:
    args = build_arg_parser(spec).parse_args()
    run(spec, args)
