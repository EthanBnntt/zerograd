"""Checkpointing and early-stopping helpers for the training examples.

Long-running examples (MNIST/CIFAR-10/ViT) can lose all progress on a crash or
interruption. These helpers add periodic checkpointing (save params + optimizer
state), resumption from a checkpoint, and early stopping on a convergence
plateau. See issue #34.

Usage::

    from _checkpoint import save_checkpoint, load_checkpoint, EarlyStopping

    es = EarlyStopping(patience=50, mode="min")
    for step in range(steps):
        ...
        if step % ckpt_interval == 0:
            save_checkpoint(ckpt_path, step, params, state)
        if es(metrics.mean_loss):
            print("early stopping: loss plateaued")
            break

    # resume:
    if args.resume:
        ck = load_checkpoint(args.resume)
        start, params, state = ck["step"] + 1, ck["params"], ck["state"]
"""

from __future__ import annotations

import os
import pickle
from typing import Any


def save_checkpoint(path: str, step: int, params: Any, state: Any, extra: Any = None) -> None:
    """Atomically pickle ``(step, params, state, extra)`` to ``path``.

    Writes to a temporary file then renames, so a crash mid-write cannot
    corrupt an existing checkpoint. Parent directories are created as needed.
    """
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump({"step": step, "params": params, "state": state, "extra": extra}, f)
    os.replace(tmp, path)


def load_checkpoint(path: str) -> dict:
    """Load a checkpoint written by :func:`save_checkpoint`."""
    with open(path, "rb") as f:
        return pickle.load(f)


class EarlyStopping:
    """Stop once a monitored metric stops improving for ``patience`` steps.

    ``mode='min'`` tracks the lowest metric (e.g. loss); ``mode='max'`` tracks
    the highest (e.g. accuracy). ``__call__`` returns ``True`` once patience is
    exhausted.
    """

    def __init__(self, patience: int = 50, mode: str = "min", min_delta: float = 0.0) -> None:
        if not isinstance(patience, int) or isinstance(patience, bool) or patience < 1:
            raise ValueError("patience must be a positive integer")
        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best: float | None = None
        self.wait = 0

    def __call__(self, metric: float) -> bool:
        if self.best is None:
            improved = True
        elif self.mode == "min":
            improved = metric < self.best - self.min_delta
        else:
            improved = metric > self.best + self.min_delta
        if improved:
            self.best = metric
            self.wait = 0
        else:
            self.wait += 1
        return self.wait >= self.patience
