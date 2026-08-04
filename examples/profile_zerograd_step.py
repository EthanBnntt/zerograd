"""Profile candidate evaluation and replay/update separately on one GPU.

This uses synthetic token batches so data loading and W&B never pollute the
timings.  Defaults match ``scripts/run_h100_minipile.sh``.

Run on the H100::

    XLA_PYTHON_CLIENT_MEM_FRACTION=.9 \
      .venv/bin/python examples/profile_zerograd_step.py
"""

from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import nnx

from zerograd import ZeroGrad


def _load_train_module():
    path = Path(__file__).with_name("train_int_rnn_minipile.py")
    spec = importlib.util.spec_from_file_location("train_int_rnn_minipile", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ready(tree) -> None:
    jax.block_until_ready(tree)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=2048)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--ffn-mult", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--vocab", type=int, default=248_077)
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--candidate-chunk", type=int, default=12)
    parser.add_argument("--ce-parallel", type=int, default=2)
    parser.add_argument("--logit-chunk", type=int, default=12_288)
    parser.add_argument("--delta-chunk", type=int, default=64)
    parser.add_argument("--gdn-feat-tile", type=int, default=32)
    args = parser.parse_args()

    train = _load_train_module()
    train.configure_architecture(
        dim=args.dim,
        heads=args.heads,
        layers=args.layers,
        ffn_mult=args.ffn_mult,
        seq_len=args.seq_len,
    )
    train.LOGIT_CHUNK = args.logit_chunk
    train.CE_PARALLEL = args.ce_parallel
    train.DELTA_CHUNK = args.delta_chunk
    train.GDN_FEAT_TILE = args.gdn_feat_tile

    print(f"devices={jax.devices()} backend={jax.default_backend()}", flush=True)
    model = train.IntRnnLM(
        args.vocab,
        rngs=nnx.Rngs(0),
        delta_impl="chunkwise",
    )
    optimizer = ZeroGrad(
        population_size=args.population,
        rank=args.rank,
        seed=0,
        run_id="profile-int-rnn",
        integer_es=True,
        sigma_shift=2,
        int_bits=8,
        update_alpha=0.12,
        alpha_decay=0.0,
        candidate_chunk_size=args.candidate_chunk,
        check_finite=False,
    )
    state = optimizer.init(model)

    key = jax.random.key(123)
    tokens = jax.random.randint(
        key,
        (args.batch, args.seq_len),
        0,
        args.vocab,
        dtype=jnp.int32,
    )
    targets = jnp.roll(tokens, -1, axis=1)
    batch = (tokens, targets)

    def loss_fn(bound_model, bound_batch):
        x, y = bound_batch
        return (
            bound_model.chunked_nll(
                x,
                y,
                chunk_size=args.logit_chunk,
                ce_parallel=args.ce_parallel,
            ),
            None,
        )

    candidate_ids = jnp.arange(args.population, dtype=jnp.int32)

    def evaluate() -> jax.Array:
        if optimizer.integer_es:
            return optimizer._evaluate_antithetical_nnx(  # noqa: SLF001
                model,
                state.generation,
                loss_fn,
                batch,
                rng=jax.random.key(0),
            )
        return optimizer.evaluate_shard(
            model,
            state.generation,
            loss_fn,
            batch,
            candidate_ids,
        )

    def update(losses):
        nonlocal model, state
        model, state, metrics = optimizer.step_from_losses(
            state,
            model,
            losses,
        )
        _ready((train.params_pure_dict(model), metrics.mean_loss))
        return metrics

    for iteration in range(2):
        t0 = time.perf_counter()
        losses = evaluate()
        _ready(losses)
        eval_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        metrics = update(losses)
        update_s = time.perf_counter() - t0

        mem = jax.local_devices()[0].memory_stats() or {}
        print(
            f"iteration={iteration} generation={state.generation} "
            f"candidate_eval_s={eval_s:.3f} replay_update_s={update_s:.3f} "
            f"total_s={eval_s + update_s:.3f} "
            f"mean_loss={float(metrics.mean_loss):.5f} "
            f"peak_gib={int(mem.get('peak_bytes_in_use', 0)) / 2**30:.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
