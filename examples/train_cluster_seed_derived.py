"""Seed-derived cluster ZeroGrad: params never communicated.

Demonstrates the core scaling property: each node computes its own
parameters from a shared seed and the sequence of fitness arrays.
Only the 1D loss array is shared between nodes — no params, gradients,
or optimizer state ever cross node boundaries.

This script runs N nodes in-process (simulating a multi-node cluster)
and verifies after every step that all nodes have identical params.
It also verifies the cluster matches a single-node baseline.

    uv run python examples/train_cluster_seed_derived.py [--steps N] [--nodes N]
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import optax

from zerograd import (
    ClusterZeroGrad,
    Manifest,
    ManifestEntry,
    ParameterLayout,
    ZeroGrad,
)

# ── Model: 2→16→1 MLP on XOR ─────────────────────────────────────────────────
INPUT_DIM = 2
HIDDEN_DIM = 16
OUTPUT_DIM = 1


def build_params(key):
    k1, k2 = jax.random.split(key)
    return {
        "w1": jax.random.normal(k1, (INPUT_DIM, HIDDEN_DIM)) * 0.5,
        "b1": jnp.zeros((HIDDEN_DIM,)),
        "w2": jax.random.normal(k2, (HIDDEN_DIM, OUTPUT_DIM)) * 0.5,
        "b2": jnp.zeros((OUTPUT_DIM,)),
    }


def build_manifest():
    return Manifest(version=1, entries=(
        ManifestEntry(("w1",), ParameterLayout.MATRIX, "w1"),
        ManifestEntry(("b1",), ParameterLayout.VECTOR, "b1"),
        ManifestEntry(("w2",), ParameterLayout.MATRIX, "w2"),
        ManifestEntry(("b2",), ParameterLayout.VECTOR, "b2"),
    ))


XOR_X = jnp.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
XOR_Y = jnp.array([[0.], [1.], [1.], [0.]])


def loss_fn(params, candidate, batch, rng):
    x, y = batch
    h = jax.nn.tanh(candidate.linear(params, ("w1",), x))
    h = h + candidate.vector(params, ("b1",))
    logits = candidate.linear(params, ("w2",), h)
    logits = logits + candidate.vector(params, ("b2",))
    return jnp.mean((logits - y) ** 2), None


def accuracy(params):
    h = jax.nn.tanh(XOR_X @ params["w1"]) + params["b1"]
    logits = h @ params["w2"] + params["b2"]
    preds = (logits > 0.5).astype(jnp.float32)
    return float(jnp.mean(preds == XOR_Y))


def main():
    parser = argparse.ArgumentParser(description="Seed-derived cluster ZeroGrad")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--nodes", type=int, default=4)
    parser.add_argument("--pop", type=int, default=32)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--sigma", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest = build_manifest()

    optimizer = ZeroGrad(
        manifest,
        optax.adamw(learning_rate=args.lr, weight_decay=0.0),
        population_size=args.pop,
        rank=args.rank,
        sigma=args.sigma,
        seed=args.seed,
        run_id="cluster-xor",
    )

    cluster = ClusterZeroGrad(
        optimizer,
        build_params,
        loss_fn,
        seed=args.seed,
        num_nodes=args.nodes,
    )

    batch = (XOR_X, XOR_Y)

    print(f"Seed-derived cluster ZeroGrad")
    print(f"  Nodes: {args.nodes} (each independently computes params from seed={args.seed})")
    print(f"  Population: {args.pop} ({cluster.partition_sizes} per node)")
    print(f"  Communication per step:")
    print(f"    Losses: {cluster.losses_bytes_per_step} bytes (shared)")
    print(f"    Params: {cluster.params_bytes} bytes (NOT shared — computed locally)")
    print(f"    Savings: {cluster.params_bytes / cluster.losses_bytes_per_step:.0f}x per step")
    if args.steps > 0:
        total_losses = cluster.losses_bytes_per_step * args.steps
        total_params_would = cluster.params_bytes * args.steps * args.nodes
        print(f"    Over {args.steps} steps: {total_losses:,} bytes losses vs "
              f"{total_params_would:,} bytes params would-be")
    print(f"  Steps: {args.steps}\n")

    # ── Baseline: single-node ──────────────────────────────────────────────────
    print("Running single-node baseline...")
    opt_single = ZeroGrad(
        manifest,
        optax.adamw(learning_rate=args.lr, weight_decay=0.0),
        population_size=args.pop,
        rank=args.rank,
        sigma=args.sigma,
        seed=args.seed,
        run_id="cluster-xor",
    )
    params_single = build_params(jax.random.key(args.seed))
    state_single = opt_single.init(params_single)
    for step in range(args.steps):
        params_single, state_single, _ = opt_single.step(
            state_single, params_single, batch, loss_fn)
    print(f"  Baseline final loss: {float(jnp.mean((params_single['w1']) ** 2)):.4f}\n")

    # ── Cluster ─────────────────────────────────────────────────────────────────
    print("Running cluster...")
    t0 = time.time()
    for step in range(args.steps):
        params, state, metrics = cluster.step(batch)
        synced = cluster.verify_sync()

        if step % 50 == 0 or step == args.steps - 1:
            acc = accuracy(params)
            print(f"  gen {metrics.generation:3d}  loss={metrics.mean_loss:.4f}  "
                  f"acc={acc:.0%}  sync={'✓' if synced else '✗'}  "
                  f"({(time.time() - t0) / (step + 1):.3f}s/step)")

    # ── Verification ────────────────────────────────────────────────────────────
    final_synced = cluster.verify_sync()
    acc = accuracy(params)

    # Check cluster matches single-node baseline
    cluster_leaves = jax.tree_util.tree_leaves(params)
    single_leaves = jax.tree_util.tree_leaves(params_single)
    max_diff = max(float(jnp.max(jnp.abs(a - b))) for a, b in zip(cluster_leaves, single_leaves))

    print(f"\n{'=' * 60}")
    print("VERIFICATION")
    print(f"{'=' * 60}")
    print(f"  All nodes have identical params:  {'✓' if final_synced else '✗'}")
    print(f"  Cluster matches single-node:      {'✓' if max_diff < 1e-5 else '✗'}  (diff={max_diff:.2e})")
    print(f"  Final accuracy:                   {acc:.0%}")
    print(f"  Total time:                       {time.time() - t0:.1f}s")
    print(f"\n  Per-step communication: {cluster.losses_bytes_per_step} bytes (losses only)")
    print(f"  Params never communicated: 0 bytes")
    print(f"  Each node computed params locally from seed + loss history")


if __name__ == "__main__":
    main()
