"""Multi-process cluster ZeroGrad: true process isolation.

Spawns N independent Python processes, each computing its own parameters
from a shared seed. Communication is via length-prefixed pickle over
stdin/stdout pipes. Only loss arrays (a few hundred bytes) are exchanged
— no params, gradients, or optimizer state ever cross process boundaries.

After training, the coordinator extracts params from each worker process
and verifies they are identical — proving that param communication was
never needed.

Each worker is a completely fresh Python process with its own JAX
context, simulating a real multi-node cluster. The only difference from
a real network cluster is the transport (pipes instead of TCP).

    uv run python examples/train_cluster_multiprocess.py [--steps N] [--nodes N]
"""

from __future__ import annotations

import argparse
import os
import pickle
import select
import struct
import subprocess
import sys
import threading
import time

import jax
import jax.numpy as jnp
import optax
import numpy as np

from zerograd import ZeroGrad, compute_partition_sizes
from _xor_model import (
    HIDDEN_DIM,
    INPUT_DIM,
    OUTPUT_DIM,
    build_manifest,
    build_params,
    loss_fn,
)


# ── IPC helpers ───────────────────────────────────────────────────────────────

def _send(obj, file):
    """Send a pickled object with a 4-byte length prefix."""
    data = pickle.dumps(obj)
    file.write(struct.pack("I", len(data)))
    file.write(data)
    file.flush()


def _recv(file):
    """Receive a length-prefixed pickled object."""
    header = _recv_exact(file, 4)
    if header is None:
        return None
    size = struct.unpack("I", header)[0]
    data = _recv_exact(file, size)
    return pickle.loads(data)


def _recv_exact(file, n):
    """Read exactly n bytes or return None on EOF."""
    buf = b""
    while len(buf) < n:
        chunk = file.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def _drain_stderr(proc, sink):
    """Background thread: drain a worker's stderr into ``sink`` (list[str]).

    Keeps the worker's stderr pipe from filling (which would block a crashed
    worker) and captures the failure message so a dead worker can be reported
    clearly instead of hanging the coordinator (see issue #19).
    """
    try:
        for line in iter(proc.stderr.readline, b""):
            sink.append(line.decode(errors="replace"))
    except Exception:
        pass


def _dead_worker_msg(proc, stderr_sink):
    tail = "".join(stderr_sink[-20:]).strip()
    msg = (f"worker process {proc.pid} exited (code {proc.returncode}) "
           f"without responding")
    if tail:
        msg += f"; stderr tail:\n{tail}"
    return msg


def _recv_checked(proc, stderr_sink, timeout=60.0):
    """Receive from a worker, raising a clear error if it has died.

    Checks ``proc.poll()`` and waits on the stdout pipe with a timeout so a
    crashed worker surfaces as an error (with its captured stderr) instead of
    blocking the coordinator forever (see issue #19).
    """
    if proc.poll() is not None:
        raise RuntimeError(_dead_worker_msg(proc, stderr_sink))
    fd = proc.stdout.fileno()
    ready, _, _ = select.select([fd], [], [], timeout)
    if not ready:
        if proc.poll() is not None:
            raise RuntimeError(_dead_worker_msg(proc, stderr_sink))
        raise TimeoutError(
            f"worker process {proc.pid} did not respond within {timeout}s"
        )
    result = _recv(proc.stdout)
    if result is None:
        # stdout hit EOF before a full message arrived — the worker has died.
        raise RuntimeError(_dead_worker_msg(proc, stderr_sink))
    return result


# ── Worker process ────────────────────────────────────────────────────────────

def run_worker(seed, pop, rank, sigma, lr):
    """Worker: computes params from seed, communicates only via losses."""
    manifest = build_manifest()
    optimizer = ZeroGrad(
        manifest,
        optax.adamw(learning_rate=lr, weight_decay=0.0),
        population_size=pop,
        rank=rank,
        sigma=sigma,
        seed=seed,
        run_id="mp-cluster-xor",
    )

    # Compute params from seed — never received from coordinator
    params = build_params(jax.random.key(seed))
    state = optimizer.init(params)

    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer

    batch = None
    while True:
        task = _recv(stdin)
        if task is None:
            break

        cmd = task[0]

        if cmd == "set_batch":
            batch = task[1]
            _send("ok", stdout)

        elif cmd == "evaluate":
            candidate_ids = jnp.array(task[1])
            gen = state.generation
            losses = optimizer.evaluate_shard(
                params, gen, loss_fn, batch, candidate_ids)
            # Send losses as numpy (tiny: pop/nodes floats)
            _send(np.array(losses, dtype=np.float32), stdout)

        elif cmd == "step":
            losses = jnp.array(task[1])
            params, state, metrics = optimizer.step_from_losses(state, params, losses)
            _send((metrics.generation, metrics.mean_loss, metrics.min_loss), stdout)

        elif cmd == "get_params":
            leaves = jax.tree_util.tree_leaves(params)
            _send([np.array(x) for x in leaves], stdout)

    _send("done", stdout)


# ── Coordinator ───────────────────────────────────────────────────────────────

def run_coordinator(nodes, steps, pop, rank, sigma, lr, seed):
    """Spawn worker processes and coordinate via pipes."""
    # Partition candidate IDs using the library's largest-remainder splitter so
    # the shards always sum to ``pop`` exactly, even when ``pop`` is not
    # divisible by ``nodes`` (see issue #18). Floor-division here previously
    # dropped the remainder candidates and made step_from_losses reject the
    # too-short loss array.
    sizes = compute_partition_sizes(pop, [1.0] * nodes)
    shard_ids = []
    offset = 0
    for size in sizes:
        shard_ids.append(np.arange(offset, offset + size, dtype=np.int32))
        offset += size

    # Spawn worker processes (copies of this script)
    workers = []
    stderr_sinks = []
    for i in range(nodes):
        proc = subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), "--worker",
             "--seed", str(seed),
             "--pop", str(pop),
             "--rank", str(rank),
             "--sigma", str(sigma),
             "--lr", str(lr)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**os.environ, "JAX_PLATFORMS": "cpu"},
        )
        sink: list[str] = []
        stderr_sinks.append(sink)
        threading.Thread(
            target=_drain_stderr, args=(proc, sink), daemon=True
        ).start()
        workers.append(proc)

    # Send batch to each worker (once — cached)
    xor_x = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]], dtype=np.float32)
    xor_y = np.array([[0.], [1.], [1.], [0.]], dtype=np.float32)
    batch = (xor_x, xor_y)

    for proc, sink in zip(workers, stderr_sinks):
        _send(("set_batch", batch), proc.stdin)
        assert _recv_checked(proc, sink) == "ok"

    # Communication accounting
    losses_bytes_per_step = pop * 4  # float32
    param_count = INPUT_DIM * HIDDEN_DIM + HIDDEN_DIM + HIDDEN_DIM * OUTPUT_DIM + OUTPUT_DIM
    params_bytes = param_count * 4

    print(f"Multi-process cluster ZeroGrad")
    print(f"  Nodes: {nodes} (separate Python processes)")
    print(f"  Population: {pop} (partition {list(sizes)} per node)")
    print(f"  Communication per step:")
    print(f"    Losses: {losses_bytes_per_step} bytes (shared via pipes)")
    print(f"    Params: {params_bytes} bytes (NOT shared — computed locally)")
    print(f"  Steps: {steps}\n")

    # Training loop
    t0 = time.time()
    for step in range(steps):
        gen = step

        # 1. Each worker evaluates its shard
        for i, proc in enumerate(workers):
            _send(("evaluate", shard_ids[i]), proc.stdin)

        all_losses = []
        for proc, sink in zip(workers, stderr_sinks):
            losses = _recv_checked(proc, sink)
            all_losses.append(losses)

        # 2. Gather losses — ONLY data crossing process boundaries
        gathered = np.concatenate(all_losses)

        # 3. Each worker independently computes the same update
        for proc in workers:
            _send(("step", gathered), proc.stdin)

        metrics_list = []
        for proc, sink in zip(workers, stderr_sinks):
            m = _recv_checked(proc, sink)
            metrics_list.append(m)

        # All workers report identical metrics
        gen, mean_loss, min_loss = metrics_list[0]

        if step % 50 == 0 or step == steps - 1:
            print(f"  gen {gen:3d}  loss={mean_loss:.4f}  "
                  f"({(time.time() - t0) / (step + 1):.3f}s/step)")

    # Extract params from each worker for verification
    print("\nExtracting params from each worker...")
    for proc in workers:
        _send(("get_params",), proc.stdin)

    worker_params = []
    for proc, sink in zip(workers, stderr_sinks):
        leaves = _recv_checked(proc, sink)
        worker_params.append(leaves)

    # Stop workers
    for proc in workers:
        _send(None, proc.stdin)
        proc.wait()

    # ── Verification ────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("VERIFICATION")
    print(f"{'=' * 60}")

    # Check all workers have identical params
    all_match = True
    for i in range(1, len(worker_params)):
        for a, b in zip(worker_params[0], worker_params[i]):
            diff = np.max(np.abs(a - b))
            if diff > 1e-5:
                print(f"  Worker 0 vs {i}: MISMATCH (diff={diff:.2e})")
                all_match = False
                break

    print(f"  All {nodes} workers have identical params: {'✓' if all_match else '✗'}")

    # Check against single-node baseline
    print("\n  Comparing against single-node baseline...")
    manifest = build_manifest()
    opt_single = ZeroGrad(
        manifest,
        optax.adamw(learning_rate=lr, weight_decay=0.0),
        population_size=pop,
        rank=rank,
        sigma=sigma,
        seed=seed,
        run_id="mp-cluster-xor",
    )
    params_s = build_params(jax.random.key(seed))
    state_s = opt_single.init(params_s)
    batch_jax = (jnp.array(xor_x), jnp.array(xor_y))
    for step in range(steps):
        params_s, state_s, _ = opt_single.step(state_s, params_s, batch_jax, loss_fn)

    single_leaves = [np.array(x) for x in jax.tree_util.tree_leaves(params_s)]
    baseline_match = True
    for a, b in zip(worker_params[0], single_leaves):
        diff = np.max(np.abs(a - b))
        if diff > 1e-5:
            print(f"  Worker vs baseline: MISMATCH (diff={diff:.2e})")
            baseline_match = False
            break

    print(f"  Cluster matches single-node baseline: {'✓' if baseline_match else '✗'}")

    # Accuracy — tree_leaves returns sorted by key: b1, b2, w1, w2
    leaves = worker_params[0]
    params_dict = {
        "b1": jnp.array(leaves[0]),
        "b2": jnp.array(leaves[1]),
        "w1": jnp.array(leaves[2]),
        "w2": jnp.array(leaves[3]),
    }
    h = jax.nn.tanh(jnp.array(xor_x) @ params_dict["w1"]) + params_dict["b1"]
    logits = h @ params_dict["w2"] + params_dict["b2"]
    preds = (logits > 0.5).astype(jnp.float32)
    acc = float(jnp.mean(preds == jnp.array(xor_y)))
    print(f"  Final accuracy: {acc:.0%}")

    total_time = time.time() - t0
    total_losses_bytes = losses_bytes_per_step * steps
    print(f"\n  Total time: {total_time:.1f}s")
    print(f"  Total communication: {total_losses_bytes:,} bytes (losses only)")
    print(f"  Params communicated: 0 bytes")
    print(f"  Each worker computed params locally from seed + loss history")


def main():
    parser = argparse.ArgumentParser(description="Multi-process cluster ZeroGrad")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--nodes", type=int, default=4)
    parser.add_argument("--pop", type=int, default=32)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--sigma", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        run_worker(args.seed, args.pop, args.rank, args.sigma, args.lr)
    else:
        run_coordinator(args.nodes, args.steps, args.pop, args.rank,
                        args.sigma, args.lr, args.seed)


if __name__ == "__main__":
    main()
