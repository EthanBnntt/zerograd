"""Chunkwise WY GDN-2 must match the serial float scan (shared int8 I/O)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def _load_train_module():
    path = Path(__file__).resolve().parents[1] / "examples" / "train_int_rnn_minipile.py"
    spec = importlib.util.spec_from_file_location("train_int_rnn_minipile", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rand_q8(key, shape, *, gate: bool = False):
    if gate:
        return jax.random.randint(key, shape, 0, 128, dtype=jnp.int32).astype(jnp.int8)
    return jax.random.randint(key, shape, -127, 128, dtype=jnp.int32).astype(jnp.int8)


class TestGdn2ChunkwiseMatchesStepwise:
    def test_outputs_are_int8_and_match(self):
        mod = _load_train_module()
        key = jax.random.key(0)
        bh, t, d = 4, 96, 32
        c = 64
        kq, kk, kv, ka, kb, kw = jax.random.split(key, 6)
        q = _rand_q8(kq, (bh, t, d))
        k = _rand_q8(kk, (bh, t, d))
        v = _rand_q8(kv, (bh, t, d))
        alpha = _rand_q8(ka, (bh, t, d), gate=True)
        erase = _rand_q8(kb, (bh, t, d), gate=True)
        write = _rand_q8(kw, (bh, t, d), gate=True)

        ys_step = mod._gdn2_stepwise_scan(q, k, v, alpha, erase, write)
        ys_chunk = mod._gdn2_chunkwise_wy(
            q, k, v, alpha, erase, write, chunk_size=c
        )

        assert ys_step.dtype == jnp.int8
        assert ys_chunk.dtype == jnp.int8
        assert ys_step.shape == (bh, t, d) == ys_chunk.shape

        diff = np.abs(
            np.asarray(ys_step, dtype=np.int16) - np.asarray(ys_chunk, dtype=np.int16)
        )
        assert int(diff.max()) <= 1
        # WY vs serial scan can disagree on ±1 rint boundaries; CUDA float32
        # pairwise ratios typically land ~1–2% of positions (still max|Δ|≤1).
        assert float((diff > 0).mean()) < 0.03

        for ys in (ys_step, ys_chunk):
            assert int(ys.min()) >= -128
            assert int(ys.max()) <= 127
            assert int(np.abs(np.asarray(ys)).max()) >= 8

    def test_matches_paper_float_step_then_quantize(self):
        """Both paths equal rint(127 · o_float) from the paper recurrence."""
        mod = _load_train_module()
        key = jax.random.key(1)
        bh, t, d = 2, 48, 16
        keys = jax.random.split(key, 6)
        q = _rand_q8(keys[0], (bh, t, d))
        k = _rand_q8(keys[1], (bh, t, d))
        v = _rand_q8(keys[2], (bh, t, d))
        alpha = _rand_q8(keys[3], (bh, t, d), gate=True)
        erase = _rand_q8(keys[4], (bh, t, d), gate=True)
        write = _rand_q8(keys[5], (bh, t, d), gate=True)

        qf, kf, vf = mod._q8_to_f_act(q), mod._q8_to_f_act(k), mod._q8_to_f_act(v)
        af = mod._q8_to_f_decay(alpha)
        bf, wf = mod._q8_to_f_gate(erase), mod._q8_to_f_gate(write)

        s = jnp.zeros((bh, d, d), dtype=jnp.float32)
        outs = []
        for i in range(t):
            s, o_f = mod._gdn2_float_step(
                s, qf[:, i], kf[:, i], vf[:, i], af[:, i], bf[:, i], wf[:, i]
            )
            outs.append(mod._quantize_gdn_out(o_f))
        ys_ref = jnp.stack(outs, axis=1)

        ys_step = mod._gdn2_stepwise_scan(q, k, v, alpha, erase, write)
        ys_chunk = mod._gdn2_chunkwise_wy(
            q, k, v, alpha, erase, write, chunk_size=16
        )

        assert np.array_equal(np.asarray(ys_step), np.asarray(ys_ref))
        diff = np.abs(
            np.asarray(ys_ref, dtype=np.int16) - np.asarray(ys_chunk, dtype=np.int16)
        )
        assert int(diff.max()) <= 1
