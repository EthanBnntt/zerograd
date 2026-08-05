"""Shared training-step evaluation, sampling, and W&B logging helpers.

``LOGIT_CHUNK`` / ``CE_PARALLEL`` are module-level knobs (like the
architecture globals in ``model.py``) so the CLI can tune them once at
startup; every helper below reads the current value at call time.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp

from zerograd._nnx import disable_candidates

from . import model as _model
from .model import IntRnnLM

# Set by the CLI; loss closes over these so ES never builds [B,T,V] logits.
LOGIT_CHUNK = 4096
CE_PARALLEL = 1

DEFAULT_PROMPT = "Once upon a time"


def nll_and_ppl(model: IntRnnLM, tokens: jax.Array, targets: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Chunked mean NLL + perplexity (VRAM-safe for ~250k vocab)."""
    nll = model.chunked_nll(
        tokens, targets, chunk_size=LOGIT_CHUNK, ce_parallel=CE_PARALLEL
    )
    return nll, jnp.exp(nll)


def loss_fn(model: IntRnnLM, batch: tuple) -> tuple[jax.Array, None]:
    tokens, targets = batch
    return (
        model.chunked_nll(
            tokens, targets, chunk_size=LOGIT_CHUNK, ce_parallel=CE_PARALLEL
        ),
        None,
    )


def generate_text(
    model: IntRnnLM,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 64,
    temperature: float = 0.8,
    seed: int = 0,
    max_ctx: int | None = None,
) -> str:
    """Autoregressive sample from the int LM (greedy if ``temperature<=0``)."""
    disable_candidates(model)
    ctx_len = int(_model.SEQ_LEN if max_ctx is None else max_ctx)
    ids = list(tokenizer.encode(prompt, add_special_tokens=False))
    if not ids:
        ids = [int(tokenizer.eos_token_id or tokenizer.pad_token_id or 0)]
    key = jax.random.key(seed)
    eos = tokenizer.eos_token_id

    for _ in range(int(max_new_tokens)):
        ctx = ids[-ctx_len:]
        tokens = jnp.asarray(ctx, dtype=jnp.int32)[None, :]
        # Chunked [1, V] — never [1, T, V].
        logits = model.last_logits(tokens, chunk_size=LOGIT_CHUNK)[0]
        key, sk = jax.random.split(key)
        if temperature is None or temperature <= 0:
            next_id = int(jnp.argmax(logits))
        else:
            scaled = logits / float(temperature)
            next_id = int(jax.random.categorical(sk, scaled))
        ids.append(next_id)
        if eos is not None and next_id == int(eos):
            break

    return tokenizer.decode(ids, skip_special_tokens=True)


def print_sample(
    model: IntRnnLM,
    tokenizer,
    *,
    generation: int,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    seed: int,
) -> None:
    t0 = time.time()
    text = generate_text(
        model,
        tokenizer,
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        seed=seed + generation,
    )
    dt = time.time() - t0
    # Single-line-ish preview; keep newlines readable.
    preview = text.replace("\n", "\\n")
    if len(preview) > 240:
        preview = preview[:240] + "…"
    print(
        f"  ┌─ sample gen={generation} ({dt:.1f}s) prompt={prompt!r}\n"
        f"  └─ {preview}",
        flush=True,
    )


def maybe_init_wandb(
    enabled: bool,
    *,
    project: str,
    name: str,
    entity: str | None = None,
    config: dict | None = None,
):
    """Start a W&B run if ``enabled``, else return ``None``.

    Shared by the single- and multi-GPU launchers so both raise the same
    helpful error when ``wandb`` is not installed.
    """
    if not enabled:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise SystemExit(
            "`--wandb` requires the wandb package.\n"
            "  uv pip install wandb"
        ) from exc
    wb = wandb.init(project=project, entity=entity, name=name, config=config or {})
    print(f"W&B run: {wb.url}", flush=True)
    return wb
