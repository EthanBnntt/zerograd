"""Simulate an unreliable decentralized cluster.

Mimics the agi@home scenario: gaming PCs that join/leave unpredictably,
with heavily asymmetric compute. Demonstrates:

  - Late joining: a new node replays loss history to catch up
  - Node downtime: pause/resume with work redistribution
  - Node death: work redistributed to survivors
  - Variable population partitioning: weights change as nodes come/go
  - Convergence: all nodes stay synced despite churn

The simulation runs a schedule of churn events and verifies that:
  1. All active nodes have identical params after each event
  2. The cluster matches a single-node baseline at the end

    uv run python examples/train_cluster_unreliable.py [--steps N]
"""

from __future__ import annotations

import argparse
import random
import time

import jax
import optax

from zerograd import FaultTolerantCluster, ZeroGrad
from _xor_model import (
    HIDDEN_DIM,
    INPUT_DIM,
    OUTPUT_DIM,
    XOR_X,
    XOR_Y,
    accuracy,
    build_manifest,
    build_params,
    loss_fn,
)


def run_baseline(steps, seed, pop, lr):
    """Single-node baseline for comparison."""
    opt = ZeroGrad(
        build_manifest(),
        optax.adamw(learning_rate=lr, weight_decay=0.0),
        population_size=pop, rank=4, sigma=0.15,
        seed=seed, run_id="unreliable-sim",
    )
    params = build_params(jax.random.key(seed))
    state = opt.init(params)
    batch = (XOR_X, XOR_Y)
    for _ in range(steps):
        params, state, _ = opt.step(state, params, batch, loss_fn)
    return params


def main():
    parser = argparse.ArgumentParser(description="Simulate unreliable decentralized cluster")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--pop", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-2)
    args = parser.parse_args()

    batch = (XOR_X, XOR_Y)

    opt = ZeroGrad(
        build_manifest(),
        optax.adamw(learning_rate=args.lr, weight_decay=0.0),
        population_size=args.pop, rank=4, sigma=0.15,
        seed=args.seed, run_id="unreliable-sim",
    )

    # Start with 2 nodes
    fc = FaultTolerantCluster(
        opt, build_params, loss_fn, seed=args.seed, initial_nodes=2,
    )

    rng = random.Random(123)

    print("Unreliable decentralized cluster simulation")
    print(f"  Model: {INPUT_DIM}→{HIDDEN_DIM}→{OUTPUT_DIM} MLP, pop={args.pop}")
    print(f"  Steps: {args.steps}")
    print(f"  Start: 2 nodes")
    print(f"  Communication: {fc.losses_bytes_per_step} bytes/step (losses only)")
    print(f"  Params: {fc.params_bytes} bytes (never communicated)")
    print()

    # Churn schedule: every N steps, randomly add/remove/pause/resume a node
    churn_interval = 20
    event_log = []

    t0 = time.time()
    for step in range(args.steps):
        gen = fc.generation

        # Random churn event
        if step > 0 and step % churn_interval == 0 and fc.num_total_nodes > 0:
            event = rng.choice(["add", "pause", "resume", "death", "weight"])

            if event == "add" and fc.num_total_nodes < 6:
                w = rng.choice([0.5, 1.0, 2.0, 4.0])
                idx = fc.add_node(weight=w, name=f"node-{fc.num_total_nodes}")
                event_log.append(f"gen {gen}: + node {idx} (weight={w})")

            elif event == "pause" and fc.num_active_nodes > 1:
                active_indices = [i for i in range(fc.num_total_nodes) if fc.get_status(i).active]
                if active_indices:
                    idx = rng.choice(active_indices)
                    fc.pause_node(idx)
                    event_log.append(f"gen {gen}: ~ pause node {idx}")

            elif event == "resume":
                paused_indices = [i for i in range(fc.num_total_nodes) if fc.get_status(i).paused]
                if paused_indices:
                    idx = rng.choice(paused_indices)
                    fc.resume_node(idx)
                    event_log.append(f"gen {gen}: > resume node {idx}")

            elif event == "death" and fc.num_total_nodes > 1:
                idx = rng.randrange(fc.num_total_nodes)
                fc.remove_node(idx)
                event_log.append(f"gen {gen}: x death node {idx}")

            elif event == "weight" and fc.num_active_nodes > 0:
                active_indices = [i for i in range(fc.num_total_nodes) if fc.get_status(i).active]
                if active_indices:
                    idx = rng.choice(active_indices)
                    w = rng.choice([0.5, 1.0, 2.0, 4.0])
                    fc.set_weight(idx, w)
                    event_log.append(f"gen {gen}: = node {idx} weight={w}")

        # Training step
        params, state, metrics = fc.step(batch)

        # Verify sync periodically
        synced = fc.verify_sync()

        if step % 20 == 0 or step == args.steps - 1:
            acc = accuracy(params)
            print(f"  gen {metrics.generation:3d}  loss={metrics.mean_loss:.4f}  "
                  f"acc={acc:.0%}  nodes={fc.num_active_nodes}/{fc.num_total_nodes}  "
                  f"part={fc.partition_sizes}  sync={'✓' if synced else '✗'}  "
                  f"({(time.time() - t0) / (step + 1):.3f}s/step)")

    # ── Print churn log ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("CHURN EVENTS")
    print(f"{'=' * 70}")
    for entry in event_log:
        print(f"  {entry}")

    # ── Verification ────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("VERIFICATION")
    print(f"{'=' * 70}")

    final_synced = fc.verify_sync()
    acc = accuracy(params)

    # Compare against single-node baseline
    baseline = run_baseline(args.steps, args.seed, args.pop, args.lr)
    baseline_match = fc.verify_against_single(baseline, atol=1e-5)

    print(f"  All nodes synced:           {'✓' if final_synced else '✗'}")
    print(f"  Matches single-node base:   {'✓' if baseline_match else '✗'}")
    print(f"  Final accuracy:             {acc:.0%}")
    print(f"  Total churn events:         {len(event_log)}")
    print(f"  Final nodes:                {fc.num_active_nodes} active / {fc.num_total_nodes} total")
    print(f"  Loss history size:          {fc.loss_history_size} generations")
    print(f"  Total time:                 {time.time() - t0:.1f}s")
    print(f"\n  Per-step communication:     {fc.losses_bytes_per_step} bytes (losses)")
    print(f"  Params communicated:        0 bytes")
    print(f"  Total losses transferred:   {fc.losses_bytes_per_step * args.steps:,} bytes")
    print(f"  Each node computed params locally from seed + loss history")


if __name__ == "__main__":
    main()
