"""Pure-int8 Gated DeltaNet-2 LM on streamed MiniPile — package form.

Split out of the former monolithic ``examples/train_int_rnn_minipile.py``
(see that file for the CLI entry point) into:

  - ``model.py``       — architecture (``IntRnnLM``, GDN-2 mixer, config)
  - ``data.py``         — tokenizer, streamed dataset, batching
  - ``checkpoint.py``   — save / read / load checkpoints
  - ``train_loop.py``   — eval, sampling, and W&B logging helpers

This ``__init__`` re-exports the symbols ``train_int_rnn_minipile_multigpu.py``
and other example scripts need, so ``from int_rnn import IntRnnLM`` (or
``from examples.int_rnn import IntRnnLM``, when imported as part of the
``examples`` package) keeps working without reaching into submodules.

Mutable runtime knobs (``model.DELTA_CHUNK``, ``model.GDN_FEAT_TILE``,
``train_loop.LOGIT_CHUNK``, ``train_loop.CE_PARALLEL``, and the architecture
globals set by ``model.configure_architecture``) are **not** re-exported as
top-level names here — re-exporting a mutable global would snapshot its
value at import time. Code that needs to change them should mutate the
submodule attribute directly, e.g. ``model.DELTA_CHUNK = 64``.
"""

from __future__ import annotations

from . import checkpoint, data, model, train_loop
from .checkpoint import load_checkpoint, read_checkpoint, save_checkpoint
from .data import (
    ByteTokenizer,
    DEFAULT_TOKENIZER,
    PrefetchBatcher,
    TokenBatcher,
    iter_minipile_token_ids,
    load_tokenizer,
)
from .model import (
    BITS,
    IntDeltaBlock,
    IntRnnLM,
    Int8Mlp,
    MultiHeadGatedDelta2Mixer,
    all_integer_params,
    configure_architecture,
    count_params,
    lut_stats,
)
from .train_loop import (
    DEFAULT_PROMPT,
    generate_text,
    loss_fn,
    maybe_init_wandb,
    nll_and_ppl,
    print_sample,
)

__all__ = [
    "checkpoint",
    "data",
    "model",
    "train_loop",
    "load_checkpoint",
    "read_checkpoint",
    "save_checkpoint",
    "ByteTokenizer",
    "DEFAULT_TOKENIZER",
    "PrefetchBatcher",
    "TokenBatcher",
    "iter_minipile_token_ids",
    "load_tokenizer",
    "BITS",
    "IntDeltaBlock",
    "IntRnnLM",
    "Int8Mlp",
    "MultiHeadGatedDelta2Mixer",
    "all_integer_params",
    "configure_architecture",
    "count_params",
    "lut_stats",
    "DEFAULT_PROMPT",
    "generate_text",
    "loss_fn",
    "maybe_init_wandb",
    "nll_and_ppl",
    "print_sample",
]
