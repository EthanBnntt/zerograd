"""Pure-int8 CIFAR-10 CNN: IntConv + IntLUT + IntLinear, trained with ZeroGrad.

Pipeline (all int8/int32 after the input cast):

  images[0,1] → int8
  IntConv → IntLUT → pool → IntConv → IntLUT → pool
  flatten → IntLinear → IntLUT → IntLinear(logits)

Int conv is im2col + int8 matmul (portable; avoids unsupported int8 cudnn/MIOpen
conv). Training uses Appendix H antithetical factors + sparse ±1 bin updates
(works for LUT + kernels without Adam lr=1 scrambling the table).

VRAM / compute: ``candidate_chunk_size`` batches ES forwards (default 16).
Raise ``--population`` and ``--candidate-chunk`` if the GPU looks idle.

    XLA_PYTHON_CLIENT_PREALLOCATE=false \\
      uv run python examples/train_cnn_int8_mlp.py --steps 200
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import optax
from _checkpoint import EarlyStopping, load_checkpoint, save_checkpoint
from _data import load_cifar10
from _integer_es_cli import (
    add_update_alpha_arg,
    require_even_population,
    resolve_candidate_chunk,
)
from flax import nnx

from zerograd import IntConv, IntLinear, IntLUT, ZeroGrad, int_avg_pool2d
from zerograd._nnx import disable_candidates, params_pure_dict, update_params

DEFAULT_STEPS = 200
DEFAULT_BATCH = 128
DEFAULT_POPULATION = 256
DEFAULT_RANK = 8
# Evaluate this many candidates at once inside lax.map (GPU occupancy).
DEFAULT_CANDIDATE_CHUNK = 16
IMG_SIZE = 32
NUM_CLASSES = 10
# Raw int32 head logits are O(1e3); scale into a CE-friendly range.
LOGIT_SCALE = 256.0

_IDENTITY_LUT = jnp.arange(-128, 128, dtype=jnp.int8)


def images_to_int8(images: jax.Array) -> jax.Array:
    """Float CIFAR ``[0, 1]`` → int8 in ``[0, 127]``."""
    x = images.reshape(-1, IMG_SIZE, IMG_SIZE, 3).astype(jnp.float32)
    return jnp.clip(jnp.rint(x * 127.0), -128, 127).astype(jnp.int8)


class Int8CnnClassifier(nnx.Module):
    """Integer CNN + learnable int8 LUT nonlinearities + int8 MLP head."""

    def __init__(self, rngs: nnx.Rngs):
        self.conv1 = IntConv(3, 32, kernel_size=3, rngs=rngs)
        self.act1 = IntLUT(init="identity", explore_shift=0, rngs=rngs)
        self.conv2 = IntConv(32, 64, kernel_size=3, rngs=rngs)
        self.act2 = IntLUT(init="identity", explore_shift=0, rngs=rngs)
        # 8×8×64 = 4096 after two stride-2 pools
        self.fc = IntLinear(4096, 128, rngs=rngs)
        self.act3 = IntLUT(init="identity", explore_shift=0, rngs=rngs)
        self.head = IntLinear(
            128, NUM_CLASSES, act_dtype=jnp.int32, rngs=rngs
        )

    def __call__(self, images: jax.Array) -> jax.Array:
        b = images.shape[0]
        x = images_to_int8(images)
        x = self.act1(self.conv1(x))
        x = int_avg_pool2d(x, (2, 2), (2, 2))
        x = self.act2(self.conv2(x))
        x = int_avg_pool2d(x, (2, 2), (2, 2))
        x = jnp.reshape(x, (b, -1))
        x = self.act3(self.fc(x))
        logits = self.head(x)
        return logits.astype(jnp.float32) / LOGIT_SCALE


def loss_fn(model, batch):
    x, y = batch
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(model(x), y)), None


def evaluate(model, x, y, *, batch_size: int = 256) -> jax.Array:
    disable_candidates(model)
    n = x.shape[0]
    correct = 0.0
    total = 0
    for start in range(0, n, batch_size):
        xb = x[start : start + batch_size]
        yb = y[start : start + batch_size]
        pred = jnp.argmax(model(xb), axis=-1)
        correct += float(jnp.sum(pred == yb))
        total += int(yb.shape[0])
    return jnp.asarray(correct / max(total, 1))


def lut_stats(model) -> tuple[int, float]:
    params = params_pure_dict(model)
    changed = 0
    total_abs = 0.0
    for name in ("act1", "act2", "act3"):
        table = params[name]["table"]
        delta = jnp.abs(table.astype(jnp.int32) - _IDENTITY_LUT.astype(jnp.int32))
        changed += int(jnp.sum(delta > 0))
        total_abs += float(jnp.mean(delta))
    return changed, total_abs / 3.0


def main():
    parser = argparse.ArgumentParser(
        description="Train pure-int8 CNN+LUT on CIFAR-10 with ZeroGrad"
    )
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--population", type=int, default=DEFAULT_POPULATION)
    parser.add_argument("--rank", type=int, default=DEFAULT_RANK)
    parser.add_argument(
        "--sigma-shift",
        type=int,
        default=2,
        help="Appendix H σ̂ (lower = larger int8 factor noise)",
    )
    add_update_alpha_arg(parser)
    parser.add_argument(
        "--candidate-chunk",
        type=int,
        default=DEFAULT_CANDIDATE_CHUNK,
        help="Candidates evaluated in parallel per lax.map batch "
        "(raise if GPU is idle; 0 = vmap entire population).",
    )
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    parser.add_argument("--early-stopping", action="store_true")
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    require_even_population(args.population)

    chunk = resolve_candidate_chunk(args.candidate_chunk)

    print(f"JAX devices: {jax.devices()}")
    print(f"Default backend: {jax.default_backend()}")

    print("Loading CIFAR-10 ...")
    x_train, y_train, x_test, y_test = load_cifar10()
    x_train = jnp.asarray(x_train)
    y_train = jnp.asarray(y_train)
    x_test = jnp.asarray(x_test)
    y_test = jnp.asarray(y_test)
    print(f"  train: {x_train.shape}, test: {x_test.shape}")

    key = jax.random.key(args.seed)
    model = Int8CnnClassifier(nnx.Rngs(args.seed))

    # Pure int8 + sparse ±1 bins (identity transform). Gentler on the LUT than
    # AdamW→snap at lr=1, which flipped nearly every table entry each step.
    optimizer = ZeroGrad(
        population_size=args.population,
        rank=args.rank,
        seed=args.seed,
        run_id="int8-cnn-cifar",
        integer_es=True,
        sigma_shift=args.sigma_shift,
        int_bits=8,
        update_alpha=args.update_alpha,
        candidate_chunk_size=chunk,
    )
    start_step = 0
    if args.resume:
        ck = load_checkpoint(args.resume)
        update_params(model, ck["params"])
        state = ck["state"]
        start_step = ck["step"] + 1
        print(f"Resumed from {args.resume} at step {start_step}")
    else:
        state = optimizer.init(model)

    assert optimizer._bin_updates, "expected pure ±1 bin updates for int8 CNN"
    params = params_pure_dict(model)
    assert all(jnp.issubdtype(v.dtype, jnp.integer) for v in jax.tree.leaves(params))
    n0, m0 = lut_stats(model)
    print(
        f"  IntConv+IntLUT+IntLinear | LUT identity Δ={n0} | "
        f"bins α={args.update_alpha} σ̂={args.sigma_shift} chunk={chunk!r}"
    )
    sample = model(x_train[:2])
    print(f"  sample logits (scaled): {sample}")

    num_train = x_train.shape[0]
    es = EarlyStopping(patience=args.patience, mode="min") if args.early_stopping else None
    print(
        f"\nTraining: {args.steps} steps, pop={args.population}, "
        f"batch={args.batch}, rank={args.rank}\n"
    )

    for step in range(start_step, args.steps):
        idx = jax.random.randint(jax.random.fold_in(key, step), (args.batch,), 0, num_train)
        batch = (jax.device_put(x_train[idx]), jax.device_put(y_train[idx]))
        t0 = time.time()
        model, state, metrics = optimizer.step(state, model, batch, loss_fn)
        jax.block_until_ready(params_pure_dict(model)["head"]["kernel"])
        dt = time.time() - t0

        if step % args.eval_every == 0 or step == args.steps - 1:
            test_acc = evaluate(model, x_test, y_test)
            n_ch, mean_d = lut_stats(model)
            print(
                f"gen {metrics.generation:3d}  "
                f"mean_loss={metrics.mean_loss:.4f}  "
                f"min_loss={metrics.min_loss:.4f}  "
                f"test_acc={float(test_acc):.1%}  "
                f"lut_Δ={n_ch}/768 (mean|{mean_d:.2f}|)  "
                f"({dt:.1f}s/step)",
                flush=True,
            )
        else:
            print(
                f"gen {metrics.generation:3d}  "
                f"mean_loss={metrics.mean_loss:.4f}  "
                f"min_loss={metrics.min_loss:.4f}  "
                f"({dt:.1f}s/step)",
                flush=True,
            )

        if args.checkpoint and (step + 1) % args.checkpoint_interval == 0:
            save_checkpoint(args.checkpoint, step, params_pure_dict(model), state)
            print(f"  checkpoint saved: {args.checkpoint}")
        if es is not None and es(float(metrics.mean_loss)):
            print(f"Early stopping at step {step}: loss plateaued for {args.patience} steps.")
            break

    n_ch, mean_d = lut_stats(model)
    print(
        f"\nFinal test accuracy: {float(evaluate(model, x_test, y_test)):.1%}  "
        f"LUT moved {n_ch}/768 from identity (mean|Δ|={mean_d:.2f})"
    )
    if args.checkpoint:
        save_checkpoint(args.checkpoint, step, params_pure_dict(model), state)


if __name__ == "__main__":
    main()
