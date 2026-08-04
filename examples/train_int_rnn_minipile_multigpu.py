"""Four-GPU, loss-only distributed ZeroGrad training for an integer GDN.

Every GPU owns a deterministic model replica. Candidate shards run concurrently;
only the population loss vector is exchanged. Each GPU replays the same update,
so no gradients or model parameters are synchronized.
"""

from __future__ import annotations

import argparse
import math
import time

import jax
import jax.numpy as jnp
from flax import nnx

import train_int_rnn_minipile as train
from int_rnn import model as rnn_model
from int_rnn import train_loop as rnn_train_loop
from zerograd import ReplicatedDistributedZeroGrad, ZeroGrad


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--devices", type=int, default=4)
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--ffn-mult", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--population", type=int, default=256)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--sigma-shift", type=int, default=4)
    parser.add_argument("--update-alpha", type=float, default=0.02)
    parser.add_argument("--alpha-decay", type=float, default=0.001)
    parser.add_argument("--candidate-chunk", type=int, default=32)
    parser.add_argument("--delta-chunk", type=int, default=64)
    parser.add_argument("--logit-chunk", type=int, default=512)
    parser.add_argument("--prefetch", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--tokenizer-mode",
        choices=("byte", "qwen"),
        default="byte",
    )
    parser.add_argument("--tokenizer", default=train.DEFAULT_TOKENIZER)
    parser.add_argument(
        "--checkpoint-out",
        default="checkpoints/int-gdn-char-4gpu-latest.pkl",
    )
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument(
        "--max-hours",
        type=float,
        default=0.0,
        help="Strict wall-clock limit including compilation (0 disables).",
    )
    parser.add_argument("--wandb-project", default="zerograd-int-gdn")
    parser.add_argument(
        "--wandb-run-name",
        default="h100-4x-gdn2-char-d1024-L4-pop256",
    )
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    if args.population % 2:
        raise SystemExit("--population must be even")
    gpu_devices = list(jax.devices("gpu"))
    if len(gpu_devices) < args.devices:
        raise SystemExit(
            f"requested {args.devices} GPUs, but JAX sees {len(gpu_devices)}: "
            f"{gpu_devices}"
        )
    devices = gpu_devices[: args.devices]

    train.configure_architecture(
        dim=args.dim,
        heads=args.heads,
        layers=args.layers,
        ffn_mult=args.ffn_mult,
        seq_len=args.seq_len,
    )
    rnn_train_loop.LOGIT_CHUNK = args.logit_chunk
    rnn_train_loop.CE_PARALLEL = 1
    rnn_model.DELTA_CHUNK = args.delta_chunk

    tokenizer = train.load_tokenizer(args.tokenizer, mode=args.tokenizer_mode)
    vocab_size = len(tokenizer)
    raw_batcher = train.TokenBatcher(
        train.iter_minipile_token_ids(tokenizer, seed=args.seed),
        batch=args.batch,
        seq_len=args.seq_len,
    )
    batcher = train.PrefetchBatcher(raw_batcher, depth=args.prefetch)
    validation = batcher.next_batch()
    warm = batcher.next_batch()

    def model_factory():
        return train.IntRnnLM(
            vocab_size,
            rngs=nnx.Rngs(args.seed),
            num_layers=args.layers,
            delta_impl="chunkwise",
        )

    def optimizer_factory():
        return ZeroGrad(
            population_size=args.population,
            rank=args.rank,
            seed=args.seed,
            run_id=f"int-rnn-minipile-{args.tokenizer_mode}-4gpu",
            integer_es=True,
            sigma_shift=args.sigma_shift,
            int_bits=train.BITS,
            update_alpha=args.update_alpha,
            alpha_decay=args.alpha_decay,
            candidate_chunk_size=args.candidate_chunk,
            check_finite=False,
        )

    run_config = {
        "devices": len(devices),
        "device_names": [str(device) for device in devices],
        "communication": "losses_only",
        "dim": args.dim,
        "heads": args.heads,
        "layers": args.layers,
        "ffn_mult": args.ffn_mult,
        "seq_len": args.seq_len,
        "batch": args.batch,
        "population": args.population,
        "rank": args.rank,
        "sigma_shift": args.sigma_shift,
        "update_alpha": args.update_alpha,
        "alpha_decay": args.alpha_decay,
        "candidate_chunk": args.candidate_chunk,
        "vocab_size": vocab_size,
        "tokenizer": args.tokenizer,
        "tokenizer_mode": args.tokenizer_mode,
        "seed": args.seed,
        "resume_from": args.resume_from,
        "max_hours": args.max_hours,
    }

    wb = rnn_train_loop.maybe_init_wandb(
        not args.no_wandb,
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=run_config,
    )

    distributed = ReplicatedDistributedZeroGrad(
        devices=devices,
        optimizer_factory=optimizer_factory,
        model_factory=model_factory,
        loss_fn=train.loss_fn,
    )
    if args.resume_from:
        payload = train.read_checkpoint(args.resume_from)
        checkpoint_config = payload["config"]
        expected = {
            "dim": args.dim,
            "heads": args.heads,
            "layers": args.layers,
            "vocab_size": vocab_size,
            "tokenizer_mode": args.tokenizer_mode,
        }
        mismatches = {
            key: (checkpoint_config.get(key), value)
            for key, value in expected.items()
            if checkpoint_config.get(key) != value
        }
        if mismatches:
            raise ValueError(f"checkpoint configuration mismatch: {mismatches}")
        distributed.restore_params(
            payload["params"],
            generation=int(payload["generation"]),
        )
        print(
            f"Resumed checkpoint {args.resume_from} at generation "
            f"{distributed.state.generation}",
            flush=True,
        )
    print(f"JAX devices: {devices}", flush=True)
    print(
        f"Model parameters: {train.count_params(distributed.model):,}; "
        f"vocab={vocab_size:,} tokenizer={args.tokenizer_mode}",
        flush=True,
    )
    print(
        f"Population partitions: {distributed.partition_sizes}; "
        f"loss communication={distributed.losses_bytes_per_step} bytes/generation",
        flush=True,
    )

    state = distributed.state
    model = distributed.model
    metrics = None
    best_nll = math.inf
    started = time.perf_counter()
    deadline = (
        started + args.max_hours * 3600.0
        if args.max_hours > 0
        else math.inf
    )
    try:
        compile_start = time.perf_counter()
        model, state, metrics = distributed.step(warm)
        compile_s = time.perf_counter() - compile_start
        with jax.default_device(devices[0]):
            val = jax.device_put(validation, devices[0])
            nll, ppl = train.nll_and_ppl(model, val[0], val[1])
            jax.block_until_ready(nll)
        best_nll = float(nll)
        print(
            f"compile={compile_s:.1f}s gen={state.generation} "
            f"nll={float(nll):.4f} ppl={float(ppl):.2f} "
            f"replicas_synced={distributed.verify_sync()}",
            flush=True,
        )
        if wb is not None:
            wb.log(
                {
                    "compile_s": compile_s,
                    "train/loss": float(metrics.mean_loss),
                    "train/pair_margin": float(metrics.mean_pair_margin),
                    "eval/nll": float(nll),
                    "eval/ppl": float(ppl),
                    "eval/best_nll": best_nll,
                },
                step=state.generation,
            )

        for _ in range(1, args.steps):
            if time.perf_counter() >= deadline:
                print(
                    f"Reached wall-clock limit of {args.max_hours:.3f} hours",
                    flush=True,
                )
                break
            batch = batcher.next_batch()
            step_start = time.perf_counter()
            model, state, metrics = distributed.step(batch)
            step_s = time.perf_counter() - step_start

            if state.generation % args.log_every == 0 or state.generation == args.steps:
                with jax.default_device(devices[0]):
                    val = jax.device_put(validation, devices[0])
                    nll, ppl = train.nll_and_ppl(model, val[0], val[1])
                    jax.block_until_ready(nll)
                nll_f, ppl_f = float(nll), float(ppl)
                best_nll = min(best_nll, nll_f)
                print(
                    f"  gen {state.generation:4d}/{args.steps} "
                    f"train={float(metrics.mean_loss):.4f} "
                    f"pairΔ={float(metrics.mean_pair_margin):.5f} "
                    f"nll={nll_f:.4f} ppl={ppl_f:.2f} best={best_nll:.4f} "
                    f"({step_s:.2f}s/gen)",
                    flush=True,
                )
                if wb is not None:
                    wb.log(
                        {
                            "train/loss": float(metrics.mean_loss),
                            "train/loss_std": float(metrics.std_loss),
                            "train/pair_margin": float(metrics.mean_pair_margin),
                            "eval/nll": nll_f,
                            "eval/ppl": ppl_f,
                            "eval/best_nll": best_nll,
                            "perf/step_s": step_s,
                            "perf/data_tokens_per_s": (
                                args.batch * args.seq_len / step_s
                            ),
                            "perf/es_tokens_per_s": (
                                args.batch
                                * args.seq_len
                                * args.population
                                / step_s
                            ),
                        },
                        step=state.generation,
                    )

            if (
                args.checkpoint_out
                and args.checkpoint_every
                and state.generation % args.checkpoint_every == 0
            ):
                path = train.save_checkpoint(
                    model,
                    args.checkpoint_out,
                    generation=state.generation,
                    config=run_config,
                )
                print(f"  checkpoint={path}", flush=True)
    finally:
        batcher.close()
        if args.checkpoint_out and metrics is not None:
            path = train.save_checkpoint(
                model,
                args.checkpoint_out,
                generation=state.generation,
                config=run_config,
            )
            print(f"Saved checkpoint: {path}", flush=True)
        distributed.shutdown()
        if wb is not None:
            wb.finish()

    print(
        f"Done generation={state.generation} best_nll={best_nll:.4f} "
        f"elapsed={(time.perf_counter() - started) / 60:.1f}m",
        flush=True,
    )


if __name__ == "__main__":
    main()
