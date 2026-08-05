"""MiniPile tokenization → packed next-token batches.

Tokenizer loading (Qwen subwords or a local byte/character vocabulary),
streamed MiniPile document iteration, and fixed-length batch packing for
``examples/train_int_rnn_minipile.py``.
"""

from __future__ import annotations

import queue
import threading
from typing import Iterator

import jax
import jax.numpy as jnp
import numpy as np

DEFAULT_TOKENIZER = "Qwen/Qwen3.6-27B"


class ByteTokenizer:
    """Reversible UTF-8 byte/character tokenizer with one document EOS token."""

    eos_token_id = 256
    pad_token_id = 256
    eos_token = "<|byte_eos|>"

    def __len__(self) -> int:
        return 257

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return list(text.encode("utf-8"))

    def decode(self, ids, *, skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        data = bytes(int(token) for token in ids if 0 <= int(token) < 256)
        return data.decode("utf-8", errors="replace")


def load_tokenizer(name: str = DEFAULT_TOKENIZER, *, mode: str = "qwen"):
    """Load Qwen subwords or the local 257-token UTF-8 byte vocabulary."""
    if mode == "byte":
        return ByteTokenizer()
    if mode != "qwen":
        raise ValueError(f"unknown tokenizer mode {mode!r}")
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Qwen tokenization needs `transformers`.\n"
            "  uv pip install transformers"
        ) from exc
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    return tok


def iter_minipile_token_ids(tokenizer, *, seed: int = 0) -> Iterator[list[int]]:
    """Yield token-id lists for streamed MiniPile docs (EOS-terminated)."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "MiniPile streaming needs `datasets`.\n"
            "  uv pip install datasets"
        ) from exc

    eos = tokenizer.eos_token_id
    if eos is None:
        eos = tokenizer.pad_token_id
    if eos is None:
        raise SystemExit("tokenizer has no eos/pad token id")

    ds = load_dataset("JeanKaddour/minipile", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=2_048)
    while True:
        for ex in ds:
            text = ex.get("text") or ""
            if not text:
                continue
            ids = tokenizer.encode(text, add_special_tokens=False)
            if not ids:
                continue
            ids.append(int(eos))
            yield ids


class TokenBatcher:
    """Pack token-id streams into fixed-length next-token batches on device."""

    def __init__(self, id_iter: Iterator[list[int]], *, batch: int, seq_len: int):
        self._it = id_iter
        self.batch = int(batch)
        self.seq_len = int(seq_len)
        self._buf: list[int] = []

    def _fill(self, n: int) -> None:
        while len(self._buf) < n:
            try:
                self._buf.extend(next(self._it))
            except StopIteration as exc:
                raise RuntimeError("token stream ended unexpectedly") from exc

    def next_batch(self) -> tuple[jax.Array, jax.Array]:
        need = self.batch * self.seq_len + 1
        self._fill(need)
        flat_np = np.asarray(self._buf[:need], dtype=np.int32)
        del self._buf[: self.batch * self.seq_len]
        starts = np.arange(self.batch) * self.seq_len
        idx = starts[:, None] + np.arange(self.seq_len + 1)[None, :]
        block = jax.device_put(jnp.asarray(flat_np[idx], dtype=jnp.int32))
        return block[:, :-1], block[:, 1:]


class PrefetchBatcher:
    """Background-thread prefetch so HF streaming never stalls the GPU step."""

    def __init__(self, batcher: TokenBatcher, *, depth: int = 4):
        self._batcher = batcher
        self._q: queue.Queue = queue.Queue(maxsize=max(1, int(depth)))
        self._err: BaseException | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                self._q.put(self._batcher.next_batch())
        except BaseException as exc:  # noqa: BLE001 — surface on next_batch
            self._err = exc
            try:
                self._q.put(None)  # unblock waiter
            except Exception:
                pass

    def next_batch(self) -> tuple[jax.Array, jax.Array]:
        item = self._q.get()
        if item is None:
            raise RuntimeError(f"prefetch failed: {self._err!r}") from self._err
        return item

    def close(self) -> None:
        self._stop.set()
