"""Shared argparse / validation bits for the integer-ES (``integer_es=True``,
±1 bin update) example scripts — currently ``train_cnn_int8_mlp.py``.

Integer-ES scripts tune the same handful of ``ZeroGrad`` knobs (population parity,
the vmap-vs-chunked candidate map, and the ±1 bin update fraction) with
identical semantics but script-specific defaults/help text elsewhere, so only
the genuinely duplicated pieces are factored out here — not a full shared
argparse builder.
"""

from __future__ import annotations

import argparse

UPDATE_ALPHA_HELP = "Fraction of int params eligible for ±1 bins (lower = gentler LUT)"


def add_update_alpha_arg(parser: argparse.ArgumentParser, *, default: float = 0.12) -> None:
    """Add the ``--update-alpha`` flag shared by every integer-ES script."""
    parser.add_argument(
        "--update-alpha",
        type=float,
        default=default,
        help=UPDATE_ALPHA_HELP,
    )


def require_even_population(population: int) -> None:
    """Raise ``SystemExit`` unless ``population`` is even (antithetical ES pairs)."""
    if population % 2 != 0:
        raise SystemExit("--population must be even (integer_es antithetical pairs)")


def resolve_candidate_chunk(candidate_chunk: int) -> int | None:
    """``--candidate-chunk 0`` means "vmap the whole population"; ``ZeroGrad``
    expects ``None`` to select that mode."""
    return None if candidate_chunk == 0 else candidate_chunk
