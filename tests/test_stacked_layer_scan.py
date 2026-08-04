"""Stacked NNX layers use scan and preserve ZeroGrad replay identity."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import nnx

from zerograd import ZeroGrad
from zerograd._factors import (
    int_matrix_factors,
    int_vector_noise,
    stacked_int_matrix_factors,
    stacked_int_vector_noise,
)


def _load_train_module():
    path = Path(__file__).resolve().parents[1] / "examples" / "train_int_rnn_minipile.py"
    spec = importlib.util.spec_from_file_location("train_int_rnn_minipile", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stacked_factors_match_folded_layer_keys():
    key = jax.random.key(7)
    layers, in_features, out_features, rank = 3, 8, 6, 2
    a, b = stacked_int_matrix_factors(
        key, (layers, in_features, out_features), rank
    )
    noise = stacked_int_vector_noise(key, (layers, out_features))

    for layer_id in range(layers):
        layer_key = jax.random.fold_in(key, layer_id)
        a_ref, b_ref = int_matrix_factors(
            layer_key, (in_features, out_features), rank
        )
        n_ref = int_vector_noise(layer_key, (out_features,))
        assert jnp.array_equal(a[layer_id], a_ref)
        assert jnp.array_equal(b[layer_id], b_ref)
        assert jnp.array_equal(noise[layer_id], n_ref)


def test_scanned_int_gdn_runs_one_integer_es_step():
    train = _load_train_module()
    train.configure_architecture(
        dim=32,
        heads=2,
        layers=3,
        ffn_mult=2,
        seq_len=8,
    )
    model = train.IntRnnLM(
        64,
        rngs=nnx.Rngs(0),
        num_layers=3,
        delta_impl="stepwise",
    )
    optimizer = ZeroGrad(
        population_size=4,
        rank=2,
        seed=0,
        run_id="stacked-scan-test",
        integer_es=True,
        candidate_chunk_size=4,
    )
    state = optimizer.init(model)

    # One manifest entry per layer role, not one copy per physical layer.
    assert len(optimizer.manifest.entries) == 20
    layer_kernel = next(
        entry
        for entry in optimizer.manifest.entries
        if entry.path == ("layers", "mixer", "in_proj", "kernel")
    )
    assert layer_kernel.layout.value == "stacked_matrix"

    tokens = jnp.arange(16, dtype=jnp.int32).reshape(2, 8)

    def loss_fn(candidate, batch):
        hidden = candidate.encode(batch)
        return jnp.mean(hidden.astype(jnp.float32) ** 2), None

    model, state, metrics = optimizer.step(state, model, tokens, loss_fn)
    assert state.generation == 1
    assert jnp.isfinite(metrics.mean_loss)
    assert model.encode(tokens).shape == (2, 8, 32)
    assert model.last_logits(tokens, chunk_size=16).shape == (2, 64)
    assert all(
        jnp.issubdtype(leaf.dtype, jnp.integer)
        for leaf in jax.tree.leaves(train.params_pure_dict(model))
    )
