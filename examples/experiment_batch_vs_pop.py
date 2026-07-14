"""ES gradient quality: batch size vs population size tradeoff.

Does the pseudo-gradient improve more from:
  A) Bigger batch (less loss variance per candidate, but same perturbation diversity)
  B) Bigger population (more perturbation directions, but same loss variance)

We control for total compute: batch_size * population_size = constant.
A single GPU evaluates all candidates, so compute ≈ pop * batch forward passes.

Configurations (pop * batch = 8192 total forward passes per step):

  1. pop=32,  batch=256   (baseline)
  2. pop=64,  batch=128   (2× population, ½ batch)
  3. pop=128, batch=64    (4× population, ¼ batch)
  4. pop=16,  batch=512   (½ population, 2× batch)
  5. pop=8,   batch=1024  (¼ population, 2× batch)

Model: 784→128→10 MLP on MNIST. 200 steps. Same seed, same lr, same sigma.

    uv run python examples/experiment_batch_vs_pop.py
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import optax

from zerograd import Manifest, ManifestEntry, ParameterLayout, ZeroGrad

from _data import load_mnist

# ── Model: 784→128→10 MLP ────────────────────────────────────────────────────
INPUT_DIM = 784
HIDDEN_DIM = 128
OUTPUT_DIM = 10


def build_params(key):
    k1, k2 = jax.random.split(key)
    return {
        "w1": jax.random.normal(k1, (INPUT_DIM, HIDDEN_DIM)) * 0.02,
        "b1": jnp.zeros((HIDDEN_DIM,)),
        "w2": jax.random.normal(k2, (HIDDEN_DIM, OUTPUT_DIM)) * 0.02,
        "b2": jnp.zeros((OUTPUT_DIM,)),
    }


def build_manifest():
    return Manifest(version=1, entries=(
        ManifestEntry(("w1",), ParameterLayout.MATRIX, "w1"),
        ManifestEntry(("b1",), ParameterLayout.VECTOR, "b1"),
        ManifestEntry(("w2",), ParameterLayout.MATRIX, "w2"),
        ManifestEntry(("b2",), ParameterLayout.VECTOR, "b2"),
    ))


def loss_fn(params, candidate, batch, rng):
    x, y = batch
    h = jax.nn.relu(candidate.linear(params, ("w1",), x))
    h = h + candidate.vector(params, ("b1",))
    logits = candidate.linear(params, ("w2",), h)
    logits = logits + candidate.vector(params, ("b2",))
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, y)), None


def evaluate(params, x_test, y_test):
    h = jax.nn.relu(x_test @ params["w1"]) + params["b1"]
    logits = h @ params["w2"] + params["b2"]
    preds = jnp.argmax(logits, axis=-1)
    return float(jnp.mean(preds == y_test))


def run_config(name, pop, batch_size, x_train, y_train, x_test, y_test, steps, sigma, lr, seed):
    manifest = build_manifest()
    params = build_params(jax.random.key(seed))

    optimizer = ZeroGrad(
        manifest,
        optax.adamw(learning_rate=lr, weight_decay=0.0),
        population_size=pop,
        rank=8,
        sigma=sigma,
        seed=seed,
        run_id=f"bvp-{name}",
    )
    state = optimizer.init(params)

    key = jax.random.key(seed + 100)
    num_train = x_train.shape[0]
    history = []

    t0 = time.time()
    for step in range(steps):
        idx = jax.random.randint(jax.random.fold_in(key, step), (batch_size,), 0, num_train)
        batch = (x_train[idx], y_train[idx])

        params, state, metrics = optimizer.step(state, params, batch, loss_fn)

        if step % 25 == 0 or step == steps - 1:
            acc = evaluate(params, x_test, y_test)
            history.append((metrics.generation, float(metrics.mean_loss), acc))
            print(f"  [{name}] gen {metrics.generation:3d}  "
                  f"loss={metrics.mean_loss:.4f}  "
                  f"acc={acc:.1%}  "
                  f"({(time.time() - t0) / (step + 1):.2f}s/step)")

    final_acc = evaluate(params, x_test, y_test)
    total_time = time.time() - t0
    return final_acc, float(metrics.mean_loss), total_time, history


def main():
    parser = argparse.ArgumentParser(description="Batch size vs population size experiment")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Loading MNIST ...")
    x_train, y_train, x_test, y_test = load_mnist()
    x_train = jnp.array(x_train)
    y_train = jnp.array(y_train)
    x_test = jnp.array(x_test)
    y_test = jnp.array(y_test)
    print(f"  train: {x_train.shape}, test: {x_test.shape}")

    total_compute = 8192  # pop * batch = constant
    configs = [
        ("pop8_b1024",    8, 1024),
        ("pop16_b512",   16,  512),
        ("pop32_b256",   32,  256),  # baseline
        ("pop64_b128",   64,  128),
        ("pop128_b64",  128,   64),
        ("pop256_b32",  256,   32),
    ]

    print(f"\nModel: {INPUT_DIM}→{HIDDEN_DIM}→{OUTPUT_DIM} MLP")
    print(f"Total compute per step: {total_compute} forward passes (pop × batch)")
    print(f"Steps: {args.steps}, sigma={args.sigma}, lr={args.lr}, seed={args.seed}")
    print(f"{'Config':<20} {'Pop':>5} {'Batch':>6} {'Compute':>8}")
    print("-" * 45)
    for name, pop, batch in configs:
        print(f"{name:<20} {pop:>5} {batch:>6} {pop*batch:>8}")
    print()

    results = {}
    for name, pop, batch in configs:
        print(f"{'=' * 60}")
        print(f"  {name}: pop={pop}, batch={batch}")
        print(f"{'=' * 60}")
        acc, loss, total_time, history = run_config(
            name, pop, batch, x_train, y_train, x_test, y_test,
            args.steps, args.sigma, args.lr, args.seed,
        )
        results[name] = {
            "pop": pop, "batch": batch,
            "final_acc": acc, "final_loss": loss,
            "total_time": total_time, "history": history,
        }
        print()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\n{'Config':<20} {'Pop':>5} {'Batch':>6} {'Acc':>8} {'Loss':>8} {'Time':>8}")
    print("-" * 58)
    for name, r in results.items():
        print(f"{name:<20} {r['pop']:>5} {r['batch']:>6} "
              f"{r['final_acc']:>7.1%} {r['final_loss']:>8.4f} {r['total_time']:>7.1f}s")

    # ── Analysis ──────────────────────────────────────────────────────────────
    print("\nKey findings:")
    best_acc = max(r["final_acc"] for r in results.values())
    best_name = [n for n, r in results.items() if r["final_acc"] == best_acc][0]
    print(f"  Best accuracy: {best_name} ({best_acc:.1%})")

    # Find inflection point
    configs_sorted = sorted(results.values(), key=lambda r: r["pop"])
    print(f"\n  Population sweep (batch decreases as pop increases):")
    for r in configs_sorted:
        bar = "█" * int(r["final_acc"] * 50)
        print(f"    pop={r['pop']:>3} batch={r['batch']:>4}: {r['final_acc']:>5.1%} {bar}")

    print(f"\n  Total compute held constant at {total_compute} forward passes/step")
    print(f"  Bigger batch → lower loss variance per candidate (cleaner fitness signal)")
    print(f"  Bigger pop → more perturbation directions (richer gradient)")


if __name__ == "__main__":
    main()
