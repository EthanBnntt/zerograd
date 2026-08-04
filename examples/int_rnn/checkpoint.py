"""Save / read / load checkpoints for ``examples/train_int_rnn_minipile.py``."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from zerograd._nnx import disable_candidates, params_pure_dict, update_params

if TYPE_CHECKING:
    from .model import IntRnnLM


def save_checkpoint(
    model: "IntRnnLM",
    path: str | Path,
    *,
    generation: int,
    config: dict,
) -> Path:
    """Atomically save trusted local model parameters and reconstruction config."""
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "zerograd-int-rnn-v1",
        "generation": int(generation),
        "config": dict(config),
        "params": jax.tree.map(lambda value: np.asarray(value), params_pure_dict(model)),
    }
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(target)
    return target


def read_checkpoint(path: str | Path) -> dict:
    """Read a trusted local checkpoint payload."""
    with Path(path).expanduser().open("rb") as handle:
        payload = pickle.load(handle)  # noqa: S301 - explicitly trusted local artifact
    if payload.get("format") != "zerograd-int-rnn-v1":
        raise ValueError("unsupported integer RNN checkpoint format")
    return payload


def load_checkpoint(model: "IntRnnLM", path: str | Path) -> dict:
    """Load a trusted checkpoint created by :func:`save_checkpoint`."""
    payload = read_checkpoint(path)
    params = jax.tree.map(jnp.asarray, payload["params"])
    update_params(model, params)
    disable_candidates(model)
    return payload
