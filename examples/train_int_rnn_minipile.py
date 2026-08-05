"""Pure-int8 Gated DeltaNet-2 LM on streamed MiniPile (next-token).

CLI entry point for the ``int_rnn`` package (``examples/int_rnn/``):

  - ``int_rnn.model``       — ``IntRnnLM`` architecture + GDN-2 mixer
  - ``int_rnn.data``        — tokenizer, streamed dataset, batching
  - ``int_rnn.checkpoint``  — save / read / load checkpoints
  - ``int_rnn.train_loop``  — eval, sampling, W&B logging helpers

Architecture (all int8 until the head). The main block is ``IntLinearLUT``:

  int8 @ int8 → int32 accum → int8 requant → learnable IntLUT

  Qwen3.6 subword ids → int8 ``IntEmbedding`` (surgery → ``ZgIntEmbedding``)
  → ×L IntDeltaBlock:
        pre-norm → multi-head **Gated Delta Rule-2** scan → residual
        pre-norm → int8 MLP (IntLinearLUT²) → residual
  → final norm → tied int8 embed gather → int32 → **float logits** → CE

Embeddings are learnable (ES ±1 bins) like ``nn.Embedding``. Table factors are
row-sparse (only gathered ids), so large Qwen vocabs do not allocate ``A[V,r]``
on every lookup. Body width is CLI-tunable for GPU compute.

Recurrent fast-weights ``S`` are fp32 (paper WY / serial scan); projections /
gates stay int8 Q8 with shared int8 requant on mixer outputs.
Gated Delta Rule-2 (Yang et al., 2026) decouples erase ``b`` and write ``w``.

H100 efficiency notes:
  - Mixer in-proj is **one** int8 GEMM ``D→6D`` (q/k/v/α/b/w) then per-slice LUTs
    so Tensor Cores see a fat IU8 matmul instead of six ``D×D`` launches.
  - Chunkwise WY factorizes per-feature decay and uses dense ``[C,D]@[D,C]``
    score GEMMs; it never materializes ``[BH,C,C,hd]`` ratio tensors.

    uv pip install datasets transformers
    # Compute-first on ~16GB (wide body, small batch×pop)
    XLA_PYTHON_CLIENT_PREALLOCATE=false uv run python examples/train_int_rnn_minipile.py \\
      --steps 2000 --dim 512 --heads 8 --layers 6 --seq-len 256 \\
      --batch 8 --population 16 --candidate-chunk 4 \\
      --delta-impl chunkwise --delta-chunk 64 --logit-chunk 8192
    # H100 80GB — 128 antithetical directions in dense candidate chunks.
    uv run python examples/train_int_rnn_minipile.py \\
      --steps 2000 --dim 2048 --heads 16 --layers 12 --seq-len 512 --ffn-mult 4 \\
      --batch 16 --population 256 --candidate-chunk 32 --ce-parallel 2 --rank 4 \\
      --sigma-shift 3 --update-alpha 0.02 --alpha-decay 0.001 \\
      --delta-impl chunkwise --delta-chunk 64 --logit-chunk 12288 --gdn-feat-tile 32
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from zerograd import ZeroGrad
from zerograd._nnx import params_pure_dict

# ``examples/`` must be on sys.path for ``int_rnn`` (and sibling single-file
# helper modules like ``_integer_es_cli``) to be importable, whether this
# file is run directly, imported as ``examples.train_int_rnn_minipile``, or
# exec'd from an arbitrary path via ``importlib.util`` (as the tests do).
_EXAMPLES_DIR = Path(__file__).resolve().parent
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from _integer_es_cli import add_update_alpha_arg, require_even_population, resolve_candidate_chunk
from int_rnn import model as rnn_model
from int_rnn import data as rnn_data
from int_rnn import checkpoint as rnn_checkpoint
from int_rnn import train_loop as rnn_train_loop

# Re-export the package's public surface so existing callers that treat this
# file as a flat module (``import train_int_rnn_minipile as train`` /
# ``importlib.util.spec_from_file_location(...)``) keep working unchanged —
# tests reach for ``train.IntRnnLM``, ``train.ByteTokenizer``,
# ``train._gdn2_chunkwise_wy``, etc. Mutable knobs (``DELTA_CHUNK``,
# ``GDN_FEAT_TILE``, ``LOGIT_CHUNK``, ``CE_PARALLEL``) are re-exported too,
# for read access; ``main()`` mutates the *submodule* attribute
# (``rnn_model.DELTA_CHUNK = ...``) rather than this snapshot, so the mixer
# and eval helpers (which live in those submodules) see the update.
from int_rnn.model import (  # noqa: F401
    BITS,
    DELTA_CHUNK,
    EGG_DIV,
    EMBED_DIM,
    FFN_DIM,
    FFN_MULT,
    GDN_FEAT_TILE,
    HEAD_DIM,
    LOGIT_SCALE,
    NUM_HEADS,
    NUM_LAYERS,
    Q8,
    SEQ_LEN,
    Int8Mlp,
    IntDeltaBlock,
    IntRnnLM,
    MultiHeadGatedDelta2Mixer,
    _as_gate,
    _decay_weighted_dots,
    _egg_add,
    _gdn2_chunkwise_wy,
    _gdn2_float_step,
    _gdn2_stepwise_scan,
    _linear_lut,
    _q8_to_f_act,
    _q8_to_f_decay,
    _q8_to_f_gate,
    _q8_to_f_qk,
    _quantize_gdn_out,
    _solve_tril_I_plus_T,
    all_integer_params,
    configure_architecture,
    count_params,
    lut_stats,
)
from int_rnn.data import (  # noqa: F401
    DEFAULT_TOKENIZER,
    ByteTokenizer,
    PrefetchBatcher,
    TokenBatcher,
    iter_minipile_token_ids,
    load_tokenizer,
)
from int_rnn.checkpoint import load_checkpoint, read_checkpoint, save_checkpoint  # noqa: F401
from int_rnn.train_loop import (  # noqa: F401
    DEFAULT_PROMPT,
    CE_PARALLEL,
    LOGIT_CHUNK,
    generate_text,
    loss_fn,
    maybe_init_wandb,
    nll_and_ppl,
    print_sample,
)


def main():
    parser = argparse.ArgumentParser(
        description="Multi-layer multi-head int8 Gated DeltaNet-2 LM on streamed MiniPile"
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument(
        "--dim",
        type=int,
        default=EMBED_DIM,
        help="Model width D (raise this for GPU compute; keep batch×pop modest)",
    )
    parser.add_argument(
        "--heads",
        type=int,
        default=NUM_HEADS,
        help="Attention heads (dim must be divisible by heads; hd>=16)",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=NUM_LAYERS,
        help="Number of IntDeltaBlock layers",
    )
    parser.add_argument(
        "--ffn-mult",
        type=int,
        default=FFN_MULT,
        help="FFN hidden size = dim * ffn_mult",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=SEQ_LEN,
        help="Packed sequence length T",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Sequence rows per step. Prefer small batch + wide --dim on 16GB.",
    )
    parser.add_argument(
        "--population",
        type=int,
        default=16,
        help="ES population (even). Large models need many antithetical directions; "
        "use --candidate-chunk to bound VRAM.",
    )
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--sigma-shift", type=int, default=2)
    add_update_alpha_arg(parser)
    parser.add_argument("--alpha-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument(
        "--candidate-chunk",
        type=int,
        default=4,
        help="Candidates per lax.map batch. 0 = vmap entire population "
        "(highest throughput; ~51GB at the H100 defaults).",
    )
    parser.add_argument(
        "--logit-chunk",
        type=int,
        default=8192,
        help="Vocab slice size for chunked CE / sampling (never builds [B,T,V]). "
        "Larger = fewer serial CE steps when VRAM allows.",
    )
    parser.add_argument(
        "--ce-parallel",
        type=int,
        default=1,
        help="Vocab tiles evaluated concurrently in chunked CE (raises SM util "
        "and VRAM). Try 4–16 on H100 80GB.",
    )
    parser.add_argument(
        "--delta-impl",
        choices=("chunkwise", "stepwise"),
        default="chunkwise",
        help="GDN-2 mixer: chunkwise=WY parallel intra-chunk (default); "
        "stepwise=token-serial integer scan",
    )
    parser.add_argument(
        "--delta-chunk",
        type=int,
        default=DELTA_CHUNK,
        help="WY chunk length C (paper uses 64)",
    )
    parser.add_argument(
        "--gdn-feat-tile",
        type=int,
        default=GDN_FEAT_TILE,
        help="Feature tile for WY decay dots (default 32; lower uses less VRAM)",
    )
    parser.add_argument(
        "--prefetch",
        type=int,
        default=4,
        help="Host-side batch queue depth (overlaps MiniPile IO with GPU)",
    )
    parser.add_argument(
        "--tokenizer-mode",
        choices=("qwen", "byte"),
        default="qwen",
        help="qwen=subword tokenizer; byte=257-token UTF-8 character/byte vocabulary",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=DEFAULT_TOKENIZER,
        help="HuggingFace tokenizer id when --tokenizer-mode=qwen",
    )
    parser.add_argument(
        "--gen-every",
        type=int,
        default=100,
        help="Generate a sample every N generations (0 disables)",
    )
    parser.add_argument(
        "--gen-tokens",
        type=int,
        default=64,
        help="New tokens to sample when generating",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="Starter prompt for periodic samples",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (0 = greedy argmax)",
    )
    parser.add_argument(
        "--update",
        choices=("bins", "adamw"),
        default="bins",
        help="bins=±1 discrete updates (default); adamw=float master+snap",
    )
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument(
        "--check-finite",
        action="store_true",
        help="Synchronize every optimizer step to raise on non-finite losses. "
        "Off by default for fully asynchronous GPU execution; warmup is always checked.",
    )
    parser.add_argument(
        "--checkpoint-out",
        type=str,
        default=None,
        help="Checkpoint path. Saves atomically at exit and on interruption.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="Also overwrite --checkpoint-out every N generations (0 disables periodic saves).",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log metrics to Weights & Biases (uses ~/.netrc / WANDB_API_KEY)",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="zerograd-int-gdn",
        help="W&B project name",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Optional W&B entity/team",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Optional W&B run name (default: auto from hyperparams)",
    )
    args = parser.parse_args()

    require_even_population(args.population)
    if args.checkpoint_every < 0:
        raise SystemExit("--checkpoint-every must be >= 0")
    if args.checkpoint_every and not args.checkpoint_out:
        raise SystemExit("--checkpoint-every requires --checkpoint-out")

    configure_architecture(
        dim=args.dim,
        heads=args.heads,
        layers=args.layers,
        ffn_mult=args.ffn_mult,
        seq_len=args.seq_len,
    )
    # Mutate the submodule attributes directly (not this file's snapshot
    # import) so the mixer (int_rnn.model) and eval helpers
    # (int_rnn.train_loop) observe the new values.
    rnn_train_loop.LOGIT_CHUNK = int(args.logit_chunk)
    rnn_model.DELTA_CHUNK = int(args.delta_chunk)
    rnn_model.GDN_FEAT_TILE = max(1, int(args.gdn_feat_tile))
    rnn_train_loop.CE_PARALLEL = max(1, int(args.ce_parallel))
    chunk = resolve_candidate_chunk(args.candidate_chunk)

    print(f"JAX devices: {jax.devices()}")
    print(f"Default backend: {jax.default_backend()}")
    print(
        f"GPU path: fused D→6D int8 in-proj + dense factorized WY; "
        f"widen --dim / --candidate-chunk / --batch; "
        f"CE logit_chunk={rnn_train_loop.LOGIT_CHUNK} ce_parallel={rnn_train_loop.CE_PARALLEL}; "
        f"candidate_chunk={chunk!r}",
        flush=True,
    )

    tokenizer_label = (
        "UTF-8 byte/character vocabulary"
        if args.tokenizer_mode == "byte"
        else repr(args.tokenizer)
    )
    print(f"Loading tokenizer {tokenizer_label} ...", flush=True)
    tokenizer = load_tokenizer(args.tokenizer, mode=args.tokenizer_mode)
    vocab_size = len(tokenizer)
    chance_nll = math.log(vocab_size)
    chance_ppl = float(vocab_size)
    print(
        f"  vocab={vocab_size:,}  eos={tokenizer.eos_token!r} "
        f"({tokenizer.eos_token_id})  pad={tokenizer.pad_token_id}",
        flush=True,
    )

    print(
        f"Model: {rnn_model.NUM_LAYERS}L multi-head Gated Delta Rule-2 "
        f"[{args.delta_impl} C={rnn_model.DELTA_CHUNK}] (IntLinearLUT) + int8 MLP | "
        f"d={rnn_model.EMBED_DIM} heads={rnn_model.NUM_HEADS} hd={rnn_model.HEAD_DIM} "
        f"ffn={rnn_model.FFN_DIM} seq={rnn_model.SEQ_LEN} vocab={vocab_size:,} "
        f"(tied learnable IntEmbedding)",
        flush=True,
    )
    print(
        f"Chance NLL={chance_nll:.3f}  chance PPL={chance_ppl:.1f}  "
        f"({args.tokenizer_mode} tokens)",
        flush=True,
    )

    print("Opening streamed MiniPile (JeanKaddour/minipile) ...", flush=True)
    raw_batcher = TokenBatcher(
        iter_minipile_token_ids(tokenizer, seed=args.seed),
        batch=args.batch,
        seq_len=rnn_model.SEQ_LEN,
    )
    batcher = PrefetchBatcher(raw_batcher, depth=args.prefetch)
    # Hold out one fixed batch so eval trends are not hidden by batch-to-batch
    # MiniPile variance. The next batch is used for compile/warmup training.
    validation = batcher.next_batch()
    warm = batcher.next_batch()
    print(
        f"  first batch tokens={warm[0].shape}  "
        f"id_range=[{int(warm[0].min())}, {int(warm[0].max())}]  "
        f"pop={args.population} chunk={chunk!r} prefetch={args.prefetch}",
        flush=True,
    )

    model = IntRnnLM(
        vocab_size,
        rngs=nnx.Rngs(args.seed),
        delta_impl=args.delta_impl,
    )
    n_params = count_params(model)

    if args.update == "bins":
        optimizer = ZeroGrad(
            population_size=args.population,
            rank=args.rank,
            seed=args.seed,
            run_id="int-rnn-minipile-bins",
            integer_es=True,
            sigma_shift=args.sigma_shift,
            int_bits=BITS,
            update_alpha=args.update_alpha,
            alpha_decay=args.alpha_decay,
            candidate_chunk_size=chunk,
            check_finite=args.check_finite,
        )
        assert optimizer.integer_es and optimizer._bin_updates
        update_mode = (
            f"±1 bins α={args.update_alpha} σ̂={args.sigma_shift} chunk={chunk!r}"
        )
    else:
        optimizer = ZeroGrad(
            optax.adamw(learning_rate=args.lr, weight_decay=0.0),
            population_size=args.population,
            rank=args.rank,
            seed=args.seed,
            run_id="int-rnn-minipile-adamw",
            integer_es=True,
            sigma_shift=args.sigma_shift,
            int_bits=BITS,
            candidate_chunk_size=chunk,
            check_finite=args.check_finite,
        )
        update_mode = f"AdamW→snap lr={args.lr} σ̂={args.sigma_shift}"

    state = optimizer.init(model)
    assert all_integer_params(model), "weights must stay integer after surgery"
    n_lut, m_lut = lut_stats(model)
    print(
        f"  params={n_params:,}  update={update_mode}  "
        f"LUT identity Δ={n_lut} mean|Δ|={m_lut:.3f}",
        flush=True,
    )

    run_config = {
        "dim": rnn_model.EMBED_DIM,
        "heads": rnn_model.NUM_HEADS,
        "head_dim": rnn_model.HEAD_DIM,
        "layers": rnn_model.NUM_LAYERS,
        "ffn_mult": rnn_model.FFN_MULT,
        "ffn_dim": rnn_model.FFN_DIM,
        "seq_len": rnn_model.SEQ_LEN,
        "batch": args.batch,
        "population": args.population,
        "rank": args.rank,
        "sigma_shift": args.sigma_shift,
        "candidate_chunk": chunk,
        "ce_parallel": rnn_train_loop.CE_PARALLEL,
        "logit_chunk": rnn_train_loop.LOGIT_CHUNK,
        "delta_impl": args.delta_impl,
        "delta_chunk": rnn_model.DELTA_CHUNK,
        "gdn_feat_tile": rnn_model.GDN_FEAT_TILE,
        "update": args.update,
        "update_alpha": args.update_alpha,
        "alpha_decay": args.alpha_decay,
        "vocab_size": vocab_size,
        "params": n_params,
        "tokenizer": args.tokenizer,
        "tokenizer_mode": args.tokenizer_mode,
        "seed": args.seed,
        "device": str(jax.devices()[0]),
    }

    run_name = args.wandb_run_name or (
        f"gdn2-d{rnn_model.EMBED_DIM}-L{rnn_model.NUM_LAYERS}-H{rnn_model.NUM_HEADS}-"
        f"B{args.batch}-P{args.population}-r{args.rank}"
    )
    wb = maybe_init_wandb(
        args.wandb,
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config=run_config,
    )

    t0 = time.time()
    model, state, metrics = optimizer.step(state, model, warm, loss_fn)
    jax.block_until_ready(metrics.mean_loss)
    if not bool(jnp.isfinite(metrics.mean_loss)):
        raise FloatingPointError("non-finite candidate loss during warmup")
    warm_nll, warm_ppl = nll_and_ppl(model, validation[0], validation[1])
    jax.block_until_ready(warm_nll)
    compile_s = time.time() - t0
    print(
        f"compile {compile_s:.1f}s  "
        f"first_loss={float(metrics.mean_loss):.4f}  "
        f"eval_nll={float(warm_nll):.4f}  ppl={float(warm_ppl):.2f}  "
        f"(chance ppl={chance_ppl:.1f})",
        flush=True,
    )
    if wb is not None:
        wb.log(
            {
                "compile_s": compile_s,
                "train/loss": float(metrics.mean_loss),
                "train/loss_std": float(metrics.std_loss),
                "train/pair_margin": float(metrics.mean_pair_margin),
                "eval/nll": float(warm_nll),
                "eval/ppl": float(warm_ppl),
                "eval/best_nll": float(warm_nll),
                "lut/changed": float(n_lut),
                "lut/mean_abs": float(m_lut),
                "chance/nll": chance_nll,
                "chance/ppl": chance_ppl,
            },
            step=int(metrics.generation),
        )
    if args.gen_every > 0:
        print_sample(
            model,
            tokenizer,
            generation=int(metrics.generation),
            prompt=args.prompt,
            max_new_tokens=args.gen_tokens,
            temperature=args.temperature,
            seed=args.seed,
        )

    history: list[dict] = []
    best_nll = float("inf")
    t_train = time.time()
    perf_window_start = t_train
    perf_window_steps = 0
    try:
        for step in range(1, args.steps):
            batch = batcher.next_batch()
            model, state, metrics = optimizer.step(state, model, batch, loss_fn)
            perf_window_steps += 1
            should_log = step % args.log_every == 0 or step == args.steps - 1
            # Only host-sync on log steps so the GPU can stay busy between prints.
            if should_log:
                jax.block_until_ready(metrics.mean_loss)
                train_window_s = time.time() - perf_window_start
                dt = train_window_s / perf_window_steps
            gen = int(metrics.generation)

            if should_log:
                nll, ppl = nll_and_ppl(model, validation[0], validation[1])
                jax.block_until_ready(nll)
                nll_f, ppl_f = float(nll), float(ppl)
                best_nll = min(best_nll, nll_f)
                elapsed = time.time() - t_train
                n_lut, m_lut = lut_stats(model)
                rec = {
                    "step": step,
                    "train_loss": float(metrics.mean_loss),
                    "loss_std": float(metrics.std_loss),
                    "pair_margin": float(metrics.mean_pair_margin),
                    "nll": nll_f,
                    "ppl": ppl_f,
                    "lut_changed": n_lut,
                    "lut_mean_abs": m_lut,
                    "dt": dt,
                    "elapsed_s": elapsed,
                }
                history.append(rec)
                print(
                    f"  gen {gen:5d}/{args.steps}  "
                    f"train={metrics.mean_loss:.4f}  "
                    f"pairΔ={metrics.mean_pair_margin:.5f}  "
                    f"nll={nll_f:.4f}  ppl={ppl_f:.2f}  "
                    f"best_nll={best_nll:.4f}  "
                    f"lutΔ={n_lut}  "
                    f"({dt:.2f}s/step, {elapsed / 60:.1f}m)",
                    flush=True,
                )
                if wb is not None:
                    tokens_per_s = (args.batch * rnn_model.SEQ_LEN) / max(dt, 1e-6)
                    wb.log(
                        {
                            "train/loss": float(metrics.mean_loss),
                            "train/loss_std": float(metrics.std_loss),
                            "train/pair_margin": float(metrics.mean_pair_margin),
                            "eval/nll": nll_f,
                            "eval/ppl": ppl_f,
                            "eval/best_nll": best_nll,
                            "lut/changed": float(n_lut),
                            "lut/mean_abs": float(m_lut),
                            "perf/step_s": dt,
                            "perf/tokens_per_s": tokens_per_s,
                            "perf/elapsed_s": elapsed,
                        },
                        step=gen,
                    )
                # Exclude eval/logging from the next asynchronous train window.
                perf_window_start = time.time()
                perf_window_steps = 0

            if args.gen_every > 0 and gen % args.gen_every == 0:
                print_sample(
                    model,
                    tokenizer,
                    generation=gen,
                    prompt=args.prompt,
                    max_new_tokens=args.gen_tokens,
                    temperature=args.temperature,
                    seed=args.seed,
                )
            if (
                args.checkpoint_out
                and args.checkpoint_every > 0
                and state.generation % args.checkpoint_every == 0
            ):
                saved = save_checkpoint(
                    model,
                    args.checkpoint_out,
                    generation=state.generation,
                    config=run_config,
                )
                print(
                    f"  checkpoint generation={state.generation} path={saved}",
                    flush=True,
                )
                # Serialization synchronizes the queue; start a fresh timing window.
                perf_window_start = time.time()
                perf_window_steps = 0
    finally:
        batcher.close()
        if args.checkpoint_out:
            saved = save_checkpoint(
                model,
                args.checkpoint_out,
                generation=state.generation,
                config=run_config,
            )
            print(
                f"Saved checkpoint generation={state.generation} path={saved}",
                flush=True,
            )
        if wb is not None:
            wb.finish()

    assert all_integer_params(model)
    if args.gen_every > 0:
        print_sample(
            model,
            tokenizer,
            generation=int(metrics.generation),
            prompt=args.prompt,
            max_new_tokens=args.gen_tokens,
            temperature=args.temperature,
            seed=args.seed,
        )
    if history:
        first, last = history[0], history[-1]
        n_lut, m_lut = lut_stats(model)
        print(
            f"\nDone. nll {first['nll']:.4f} → {last['nll']:.4f}  "
            f"ppl {first['ppl']:.2f} → {last['ppl']:.2f}  "
            f"LUT Δ={n_lut} mean|Δ|={m_lut:.3f}  "
            f"(chance ppl={chance_ppl:.1f})",
            flush=True,
        )


if __name__ == "__main__":
    main()
