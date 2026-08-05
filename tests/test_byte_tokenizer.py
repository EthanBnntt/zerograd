"""UTF-8 byte/character tokenization for the diagnostic LM run."""

from __future__ import annotations

import importlib.util
import itertools
from pathlib import Path

import jax
import numpy as np
from flax import nnx

from zerograd import ZeroGrad


def _load_train_module():
    path = Path(__file__).resolve().parents[1] / "examples" / "train_int_rnn_minipile.py"
    spec = importlib.util.spec_from_file_location("train_int_rnn_minipile", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_byte_tokenizer_round_trips_utf8():
    train = _load_train_module()
    tokenizer = train.ByteTokenizer()
    text = "GDN learns: café λ"
    ids = tokenizer.encode(text, add_special_tokens=False)
    assert len(tokenizer) == 257
    assert ids and all(0 <= token < 256 for token in ids)
    assert tokenizer.decode(ids, skip_special_tokens=True) == text
    assert tokenizer.decode(ids + [tokenizer.eos_token_id]) == text


def test_byte_token_batch_is_shifted_next_token_data():
    train = _load_train_module()
    tokenizer = train.ByteTokenizer()
    document = tokenizer.encode("abcdef") + [tokenizer.eos_token_id]
    batcher = train.TokenBatcher(
        itertools.cycle((document,)),
        batch=2,
        seq_len=4,
    )
    tokens, targets = batcher.next_batch()
    assert tokens.shape == targets.shape == (2, 4)
    np.testing.assert_array_equal(np.asarray(tokens)[:, 1:], np.asarray(targets)[:, :-1])
    assert int(tokens.min()) >= 0
    assert int(targets.max()) < len(tokenizer)


def test_integer_rnn_checkpoint_round_trip(tmp_path):
    train = _load_train_module()
    train.configure_architecture(
        dim=32,
        heads=2,
        layers=2,
        ffn_mult=2,
        seq_len=8,
    )

    def create(seed):
        model = train.IntRnnLM(
            257,
            rngs=nnx.Rngs(seed),
            num_layers=2,
            delta_impl="stepwise",
        )
        optimizer = ZeroGrad(
            population_size=4,
            rank=2,
            seed=seed,
            integer_es=True,
            candidate_chunk_size=4,
        )
        optimizer.init(model)
        return model

    source = create(0)
    path = tmp_path / "character-model.pkl"
    train.save_checkpoint(source, path, generation=17, config={"tokenizer_mode": "byte"})

    restored = create(1)
    payload = train.load_checkpoint(restored, path)
    assert payload["generation"] == 17
    assert payload["config"]["tokenizer_mode"] == "byte"

    source_leaves = jax.tree.leaves(train.params_pure_dict(source))
    restored_leaves = jax.tree.leaves(train.params_pure_dict(restored))
    assert len(source_leaves) == len(restored_leaves)
    for expected, actual in zip(source_leaves, restored_leaves, strict=True):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
