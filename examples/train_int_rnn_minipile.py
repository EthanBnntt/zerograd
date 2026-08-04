"""Pure-int8 Gated DeltaNet-2 LM on streamed MiniPile (next-token).

Architecture (all int8 until the head). The main block is ``IntLinearLUT``:

  int8 @ int8 → int32 accum → int8 requant → learnable IntLUT

  Qwen3.6 subword ids → int8 ``nnx.Embed`` (surgery → ``ZgEmbed``)
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
import pickle
import queue
import threading
import time
from pathlib import Path
from typing import Iterator

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from zerograd import (
    IntAffine,
    IntLinear,
    IntLinearLUT,
    IntLUT,
    ZeroGrad,
    egg_clip_cast,
)
from zerograd._integer import egg_init_matrix, egg_matmul_divisor, int_matmul
from zerograd._nnx import LayerIndex, disable_candidates, params_pure_dict, update_params

# ── Architecture defaults (overridden by CLI via ``configure_architecture``) ─
DEFAULT_TOKENIZER = "Qwen/Qwen3.6-27B"
EMBED_DIM = 128
NUM_LAYERS = 4
NUM_HEADS = 4
HEAD_DIM = EMBED_DIM // NUM_HEADS
FFN_DIM = 512
FFN_MULT = 4
SEQ_LEN = 128
BITS = 8
# Q8 scale: gates / products use / 127 (≈ float gate in [0, 1]).
Q8 = 127
# Tied-embed attend applies ``// (16√D)`` once. The resulting integer logits
# live on the Q8 activation scale, so divide by 127 rather than repeating the
# width normalization (which made the softmax nearly uniform).
LOGIT_SCALE = float(Q8)
EGG_DIV = egg_matmul_divisor(EMBED_DIM)
# Paper default chunk length for WY intra-chunk parallel (scan only across chunks).
DELTA_CHUNK = 64
# Feature tile for memory-light pairwise decay dots (avoids [BH,C,C,hd] buffers).
GDN_FEAT_TILE = 32
Q8_F = float(Q8)
_EPS = 1e-6
_LOG_CLIP = 60.0
# Keep recurrent retention high enough that a whole WY chunk has a stable,
# factorable decay range. Erase/write gates still span (0, 1].
_ALPHA_FLOOR = 0.95
_IDENTITY_LUT = jnp.arange(-128, 128, dtype=jnp.int8)


def configure_architecture(
    *,
    dim: int,
    heads: int,
    layers: int,
    ffn_mult: int,
    seq_len: int,
) -> None:
    """Set module-level width/depth used by model constructors."""
    global EMBED_DIM, NUM_LAYERS, NUM_HEADS, HEAD_DIM, FFN_DIM, FFN_MULT, SEQ_LEN
    global LOGIT_SCALE, EGG_DIV

    if dim < 32 or dim % heads != 0:
        raise SystemExit(f"--dim must be >=32 and divisible by --heads (got {dim}, {heads})")
    hd = dim // heads
    if hd < 16:
        raise SystemExit(f"head_dim=dim/heads must be >=16 (got {hd})")
    if layers < 1:
        raise SystemExit("--layers must be >=1")
    if ffn_mult < 1:
        raise SystemExit("--ffn-mult must be >=1")
    if seq_len < 8:
        raise SystemExit("--seq-len must be >=8")

    EMBED_DIM = int(dim)
    NUM_HEADS = int(heads)
    HEAD_DIM = hd
    NUM_LAYERS = int(layers)
    FFN_MULT = int(ffn_mult)
    FFN_DIM = EMBED_DIM * FFN_MULT
    SEQ_LEN = int(seq_len)
    LOGIT_SCALE = float(Q8)
    EGG_DIV = egg_matmul_divisor(EMBED_DIM)


def _egg_add(a: jax.Array, b: jax.Array) -> jax.Array:
    return egg_clip_cast(a.astype(jnp.int32) + b.astype(jnp.int32))


def _linear_lut(in_features: int, out_features: int, *, rngs: nnx.Rngs) -> IntLinearLUT:
    """Main int8 block: GEMM + identity-init LUT."""
    return IntLinearLUT(
        in_features,
        out_features,
        use_bias=False,
        bits=BITS,
        lut_init="identity",
        explore_shift=0,
        rngs=rngs,
    )


def _as_gate(x: jax.Array) -> jax.Array:
    """Map signed int8 LUT output → unsigned Q8 gate in ``[0, 127]``."""
    return ((x.astype(jnp.int32) + Q8) // 2)


# ── Multi-head integer Gated DeltaNet-2 token mixer ───────────────────────────


def _solve_tril_I_plus_T(T: jax.Array) -> jax.Array:
    """A = (I + T)^{-1} for strictly lower-triangular T (batched)."""
    eye = jnp.eye(T.shape[-1], dtype=T.dtype)
    eye = jnp.broadcast_to(eye, T.shape)
    return jax.lax.linalg.triangular_solve(
        eye + T, eye, left_side=True, lower=True, unit_diagonal=False
    )


def _q8_to_f_act(x: jax.Array) -> jax.Array:
    return x.astype(jnp.float32) / Q8_F


def _q8_to_f_qk(x: jax.Array) -> jax.Array:
    """Q/K conversion with standard head-dimension normalization."""
    return _q8_to_f_act(x) / math.sqrt(float(x.shape[-1]))


def _q8_to_f_gate(x: jax.Array) -> jax.Array:
    """Unsigned Q8 → (0, 1] for α / erase / write gates."""
    return jnp.clip(x.astype(jnp.float32) / Q8_F, _EPS, 1.0)


def _q8_to_f_decay(x: jax.Array) -> jax.Array:
    """Unsigned Q8 → high-retention recurrent decay in ``[0.95, 1]``."""
    gate = jnp.clip(x.astype(jnp.float32) / Q8_F, 0.0, 1.0)
    return _ALPHA_FLOOR + (1.0 - _ALPHA_FLOOR) * gate


def _quantize_gdn_out(o_f: jax.Array) -> jax.Array:
    """Float mixer output → int8 (shared by stepwise and chunkwise)."""
    return egg_clip_cast(jnp.rint(o_f * Q8_F))


def _gdn2_float_step(
    s_prev: jax.Array,
    q_t: jax.Array,
    k_t: jax.Array,
    v_t: jax.Array,
    alpha_t: jax.Array,
    b_t: jax.Array,
    w_t: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """One Gated Delta Rule-2 step in float32 (paper Eq. 29).

    S_t = (I - k e^T) diag(α) S_{t-1} + k z^T,  e=b⊙k, z=w⊙v,  o=S_t^T q
    """
    e = b_t * k_t
    z = w_t * v_t
    s_bar = s_prev * alpha_t[:, :, None]
    r = jnp.einsum("bkd,bk->bd", s_bar, e)
    s_new = s_bar + k_t[:, :, None] * (z - r)[:, None, :]
    o = jnp.einsum("bkd,bk->bd", s_new, q_t)
    return s_new, o


def _gdn2_stepwise_scan(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    alpha: jax.Array,
    erase: jax.Array,
    write: jax.Array,
) -> jax.Array:
    """Token-serial GDN-2: same float recurrence as WY, int8 in/out."""
    bh, t, d = q.shape
    qf, kf, vf = _q8_to_f_qk(q), _q8_to_f_qk(k), _q8_to_f_act(v)
    af = _q8_to_f_decay(alpha)
    bf, wf = _q8_to_f_gate(erase), _q8_to_f_gate(write)
    step_in = jnp.stack(
        [
            jnp.swapaxes(qf, 0, 1),
            jnp.swapaxes(kf, 0, 1),
            jnp.swapaxes(vf, 0, 1),
            jnp.swapaxes(af, 0, 1),
            jnp.swapaxes(bf, 0, 1),
            jnp.swapaxes(wf, 0, 1),
        ],
        axis=2,
    )
    s0 = jnp.zeros((bh, d, d), dtype=jnp.float32)

    def step(s_prev: jax.Array, inp: jax.Array):
        s_new, o_f = _gdn2_float_step(
            s_prev, inp[:, 0], inp[:, 1], inp[:, 2], inp[:, 3], inp[:, 4], inp[:, 5]
        )
        return s_new, _quantize_gdn_out(o_f)

    _s_final, ys = jax.lax.scan(step, s0, step_in)
    return jnp.swapaxes(ys, 0, 1)


def _decay_weighted_dots(
    left: jax.Array,
    right: jax.Array,
    log_g: jax.Array,
    *,
    tril_k: int,
    feat_tile: int,
) -> jax.Array:
    """Dense decay-weighted score GEMM without ``[BH,C,C,D]`` ratios.

    ``exp(G_r-G_s)`` factorizes as ``exp(G_r)·exp(-G_s)``.  The high-retention
    alpha parameterization keeps a 64-token chunk's exponent range small, so
    two scaled ``[C,D]`` operands feed one dense Tensor Core-friendly GEMM.
    """
    del feat_tile  # retained as a backwards-compatible CLI argument
    gamma = jnp.exp(log_g)
    inv_gamma = jnp.exp(-log_g)
    lhs = left * gamma
    rhs = right * inv_gamma
    # WY's triangular solve amplifies TF32 score error; request true fp32
    # accumulation here while retaining the dense batched GEMM.
    scores = jnp.matmul(
        lhs,
        jnp.swapaxes(rhs, -1, -2),
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.tril(scores, k=tril_k)


def _gdn2_chunkwise_wy(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    alpha: jax.Array,
    erase: jax.Array,
    write: jax.Array,
    *,
    chunk_size: int = DELTA_CHUNK,
    feat_tile: int = GDN_FEAT_TILE,
) -> jax.Array:
    """Chunkwise WY form of the same float GDN-2 recurrence (paper App. A).

    Uses stable pairwise ratios ``exp(G_r - G_s)`` instead of forming
    ``k / γ`` (which underflows for long chunks). Equivalent to
    ``_gdn2_stepwise_scan`` up to rare ±1 int8 rint boundaries.

    Decay-weighted ``C×C`` scores are accumulated over feature tiles so the
    huge ``[BH,C,C,hd]`` ratio tensor is never allocated — the dominant VRAM
    spike in the previous implementation.
    """
    bh, t, d = q.shape
    c = int(chunk_size)
    pad = (c - (t % c)) % c
    if pad:
        zpad = ((0, 0), (0, pad), (0, 0))
        q = jnp.pad(q, zpad)
        k = jnp.pad(k, zpad)
        v = jnp.pad(v, zpad)
        alpha = jnp.pad(alpha, zpad, constant_values=Q8)
        erase = jnp.pad(erase, zpad)
        write = jnp.pad(write, zpad)
    t_pad = t + pad
    n_chunks = t_pad // c

    def _chunkify(x: jax.Array) -> jax.Array:
        return x.reshape(bh, n_chunks, c, d)

    q_n = jnp.swapaxes(_chunkify(q), 0, 1)
    k_n = jnp.swapaxes(_chunkify(k), 0, 1)
    v_n = jnp.swapaxes(_chunkify(v), 0, 1)
    a_n = jnp.swapaxes(_chunkify(alpha), 0, 1)
    b_n = jnp.swapaxes(_chunkify(erase), 0, 1)
    w_n = jnp.swapaxes(_chunkify(write), 0, 1)

    s0 = jnp.zeros((bh, d, d), dtype=jnp.float32)
    tile = max(1, int(feat_tile))

    def chunk_step(s_prev: jax.Array, inputs: tuple):
        q_ch, k_ch, v_ch, a_ch, b_ch, w_ch = inputs
        qf, kf, vf = _q8_to_f_qk(q_ch), _q8_to_f_qk(k_ch), _q8_to_f_act(v_ch)
        af = _q8_to_f_decay(a_ch)
        bf, wf = _q8_to_f_gate(b_ch), _q8_to_f_gate(w_ch)

        e = bf * kf
        z = wf * vf
        log_g = jnp.cumsum(jnp.log(af), axis=1)
        t_mat = _decay_weighted_dots(e, kf, log_g, tril_k=-1, feat_tile=tile)
        a_mat = _solve_tril_I_plus_T(t_mat)

        gamma = jnp.exp(jnp.clip(log_g, -_LOG_CLIP, 0.0))
        gamma_c = gamma[:, -1, :]
        y_mat = jnp.matmul(a_mat, gamma * e)
        u_mat = jnp.matmul(a_mat, z)
        r_mat = u_mat - jnp.matmul(y_mat, s_prev)

        a_qk = _decay_weighted_dots(qf, kf, log_g, tril_k=0, feat_tile=tile)
        o_f = jnp.matmul(gamma * qf, s_prev) + jnp.matmul(a_qk, r_mat)

        k_tail = (
            jnp.exp(jnp.clip(log_g[:, -1:, :] - log_g, -_LOG_CLIP, _LOG_CLIP)) * kf
        )
        s_new = gamma_c[:, :, None] * s_prev + jnp.matmul(
            jnp.swapaxes(k_tail, -1, -2), r_mat
        )
        return s_new, _quantize_gdn_out(o_f)

    _s_final, ys = jax.lax.scan(chunk_step, s0, (q_n, k_n, v_n, a_n, b_n, w_n))
    ys = jnp.swapaxes(ys, 0, 1).reshape(bh, t_pad, d)
    return ys[:, :t, :]


class MultiHeadGatedDelta2Mixer(nnx.Module):
    """Multi-head Gated Delta Rule-2: chunkwise-WY (default) or stepwise scan.

    Input projections ``q/k/v/α/b/w`` share one int8 GEMM ``D→6D`` (H100-friendly
    IU8 Tensor Core shape) then per-branch ``IntLUT``s. Intra-chunk uses the
    paper WY form in float32 with factorized dense score GEMMs; inter-chunk
    scan carries state. Gates stay int8 Q8.
    """

    def __init__(self, *, rngs: nnx.Rngs, impl: str = "chunkwise"):
        self.impl = impl
        # One fat int8 matmul beats six D×D launches for Tensor Core occupancy.
        self.in_proj = IntLinear(
            EMBED_DIM,
            6 * EMBED_DIM,
            use_bias=False,
            bits=BITS,
            act_dtype=jnp.int8,
            rngs=rngs,
        )
        self.q_lut = IntLUT(init="identity", explore_shift=0, rngs=rngs)
        self.k_lut = IntLUT(init="identity", explore_shift=0, rngs=rngs)
        self.v_lut = IntLUT(init="identity", explore_shift=0, rngs=rngs)
        self.alpha_lut = IntLUT(init="identity", explore_shift=0, rngs=rngs)
        self.b_lut = IntLUT(init="identity", explore_shift=0, rngs=rngs)
        self.w_lut = IntLUT(init="identity", explore_shift=0, rngs=rngs)
        self.o = _linear_lut(EMBED_DIM, EMBED_DIM, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        bsz, t, _ = x.shape
        # [B,T,6D] int8 after EGG requant; LUTs stay branch-specific.
        proj = self.in_proj(x)
        chunks = jnp.split(proj, 6, axis=-1)
        q = self.q_lut(chunks[0]).reshape(bsz, t, NUM_HEADS, HEAD_DIM)
        k = self.k_lut(chunks[1]).reshape(bsz, t, NUM_HEADS, HEAD_DIM)
        v = self.v_lut(chunks[2]).reshape(bsz, t, NUM_HEADS, HEAD_DIM)
        alpha = _as_gate(self.alpha_lut(chunks[3])).reshape(bsz, t, NUM_HEADS, HEAD_DIM)
        erase = _as_gate(self.b_lut(chunks[4])).reshape(bsz, t, NUM_HEADS, HEAD_DIM)
        write = _as_gate(self.w_lut(chunks[5])).reshape(bsz, t, NUM_HEADS, HEAD_DIM)

        def _fold(a: jax.Array) -> jax.Array:
            return a.transpose(0, 2, 1, 3).reshape(bsz * NUM_HEADS, t, HEAD_DIM)

        q_f, k_f, v_f = _fold(q), _fold(k), _fold(v)
        a_f, e_f, w_f = _fold(alpha), _fold(erase), _fold(write)

        if self.impl == "stepwise":
            ys = _gdn2_stepwise_scan(q_f, k_f, v_f, a_f, e_f, w_f)
        else:
            ys = _gdn2_chunkwise_wy(
                q_f,
                k_f,
                v_f,
                a_f,
                e_f,
                w_f,
                chunk_size=DELTA_CHUNK,
                feat_tile=GDN_FEAT_TILE,
            )

        ys = ys.reshape(bsz, NUM_HEADS, t, HEAD_DIM).transpose(0, 2, 1, 3)
        return self.o(ys.reshape(bsz, t, EMBED_DIM))


class Int8Mlp(nnx.Module):
    """Pure int8 MLP as two ``IntLinearLUT`` blocks (matmul→LUT→matmul→LUT)."""

    def __init__(self, *, rngs: nnx.Rngs):
        self.fc1 = _linear_lut(EMBED_DIM, FFN_DIM, rngs=rngs)
        self.fc2 = _linear_lut(FFN_DIM, EMBED_DIM, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.fc2(self.fc1(x))


class IntDeltaBlock(nnx.Module):
    """One pre-norm DeltaNet-2 block: mixer residual + MLP residual."""

    def __init__(self, *, rngs: nnx.Rngs, delta_impl: str = "chunkwise"):
        self.n1 = IntAffine(EMBED_DIM, bits=BITS, rngs=rngs)
        self.mixer = MultiHeadGatedDelta2Mixer(rngs=rngs, impl=delta_impl)
        self.n2 = IntAffine(EMBED_DIM, bits=BITS, rngs=rngs)
        self.mlp = Int8Mlp(rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = _egg_add(x, self.mixer(self.n1(x)))
        return _egg_add(x, self.mlp(self.n2(x)))


_LAYER_SCAN_AXES = nnx.StateAxes(
    (
        (nnx.Param, 0),
        (LayerIndex, 0),
        # Shared ZeroGradSlot key/sign Variables are broadcast across layers.
        (True, None),
    )
)


@nnx.scan(
    in_axes=(nnx.Carry, _LAYER_SCAN_AXES),
    out_axes=nnx.Carry,
    graph=True,
)
def _scan_delta_layer(x: jax.Array, layer: IntDeltaBlock) -> jax.Array:
    """Apply stacked layers sequentially as one XLA scan."""
    return layer(x)


def _egg_embed_init(rng: jax.Array, shape: tuple[int, ...], dtype: jnp.dtype) -> jax.Array:
    """EGG int8 embedding init for ``nnx.Embed``."""
    del dtype
    return egg_init_matrix(rng, shape)


class IntRnnLM(nnx.Module):
    """``NUM_LAYERS``-deep pure-integer causal LM (Gated DeltaNet-2 + int8 MLP)."""

    def __init__(
        self,
        vocab_size: int,
        *,
        rngs: nnx.Rngs,
        num_layers: int = NUM_LAYERS,
        delta_impl: str = "chunkwise",
    ):
        self.vocab_size = int(vocab_size)
        self.delta_impl = delta_impl
        # nnx.Embed → surgery → ZgEmbed (learnable TABLE, row-sparse ES factors).
        self.embed = nnx.Embed(
            num_embeddings=self.vocab_size,
            features=EMBED_DIM,
            dtype=jnp.int8,
            param_dtype=jnp.int8,
            embedding_init=_egg_embed_init,
            rngs=rngs,
        )
        layers = int(num_layers)

        @nnx.split_rngs(splits=layers)
        @nnx.vmap(in_axes=(0,), out_axes=0)
        def create_layer(layer_rngs: nnx.Rngs):
            return IntDeltaBlock(rngs=layer_rngs, delta_impl=delta_impl)

        # Parameters carry one leading layer axis and are consumed by
        # ``_scan_delta_layer``. This avoids Python-unrolling twelve copies of
        # the block into the candidate executable.
        self.layers = create_layer(rngs)
        self.norm_f = IntAffine(EMBED_DIM, bits=BITS, rngs=rngs)

    def _prepare_embed_factors(self) -> tuple[jax.Array, jax.Array] | None:
        prepare = getattr(self.embed, "candidate_factors", None)
        return prepare() if prepare is not None else None

    def _embed_rows(
        self,
        indices: jax.Array,
        factors: tuple[jax.Array, jax.Array] | None,
    ) -> jax.Array:
        lookup = getattr(self.embed, "lookup", None)
        if lookup is not None:
            return lookup(indices, factors=factors)
        return self.embed(indices)

    def encode(
        self,
        tokens: jax.Array,
        *,
        embed_factors: tuple[jax.Array, jax.Array] | None = None,
    ) -> jax.Array:
        """Token ids → int8 hidden states ``[B, T, D]`` (no vocab projection)."""
        x = self._embed_rows(tokens, embed_factors)
        # No positional embeddings — recurrence carries order.
        return self.norm_f(_scan_delta_layer(x, self.layers))

    def _logits_chunk(
        self,
        h: jax.Array,
        start: int,
        size: int,
        *,
        embed_factors: tuple[jax.Array, jax.Array] | None = None,
    ) -> jax.Array:
        """Float logits for vocab slice ``[start, start+size)``; pad past ``V`` with -inf."""
        v = self.vocab_size
        idx = start + jnp.arange(size)
        valid = idx < v
        idx_c = jnp.minimum(idx, jnp.maximum(v - 1, 0))
        # Gather (factor-perturbed under ES) then int8×int8→int32 attend.
        e_c = self._embed_rows(idx_c, embed_factors)  # [C, D]
        logits_i32 = int_matmul(h, e_c.T) // EGG_DIV
        logits = logits_i32.astype(jnp.float32) / LOGIT_SCALE
        neg_inf = jnp.asarray(-1.0e9, dtype=jnp.float32)
        return jnp.where(valid, logits, neg_inf)

    def last_logits(self, tokens: jax.Array, *, chunk_size: int = 4096) -> jax.Array:
        """Float logits for the last position only ``[B, V]`` (chunked, VRAM-safe)."""
        embed_factors = self._prepare_embed_factors()
        h = self.encode(tokens, embed_factors=embed_factors)[:, -1:, :]  # [B, 1, D]
        v = self.vocab_size
        n_chunks = (v + chunk_size - 1) // chunk_size

        def one_chunk(chunk_id):
            return self._logits_chunk(
                h,
                chunk_id * chunk_size,
                chunk_size,
                embed_factors=embed_factors,
            )[:, 0, :]

        # Vocabulary chunks are independent; map them in one compiled program
        # and trim the padded tail rather than Python-looping over launches.
        parts = jax.lax.map(
            one_chunk,
            jnp.arange(n_chunks, dtype=jnp.int32),
        )
        return parts.transpose(1, 0, 2).reshape(h.shape[0], -1)[:, :v]

    def chunked_nll(
        self,
        tokens: jax.Array,
        targets: jax.Array,
        *,
        chunk_size: int = 4096,
        ce_parallel: int = 1,
    ) -> jax.Array:
        """Mean token NLL without ever materializing ``[B, T, V]`` logits.

        ``ce_parallel>1`` evaluates multiple vocab tiles concurrently (higher
        SM util / VRAM) then reduces with ``logsumexp``.
        """
        # Input, target, and every vocabulary chunk share one exact A/B draw.
        # Previously each gather regenerated A[V,r], dominating large-vocab ES.
        embed_factors = self._prepare_embed_factors()
        h = self.encode(tokens, embed_factors=embed_factors)  # [B, T, D]
        v = self.vocab_size
        n_chunks = (v + chunk_size - 1) // chunk_size
        chunk_ids = jnp.arange(n_chunks, dtype=jnp.int32)

        def one_chunk(i):
            start = i * chunk_size
            logits = self._logits_chunk(
                h, start, chunk_size, embed_factors=embed_factors
            )
            return jax.nn.logsumexp(logits, axis=-1)

        par = int(ce_parallel)
        if par <= 1:
            lse_parts = jax.lax.map(one_chunk, chunk_ids)
        else:
            lse_parts = jax.lax.map(one_chunk, chunk_ids, batch_size=par)
        lse = jax.nn.logsumexp(lse_parts, axis=0)
        # Target logit via gathered embed row (same factors as chunk path).
        e_t = self._embed_rows(targets, embed_factors)  # [B, T, D]
        t_logit = (
            jnp.sum(h.astype(jnp.int32) * e_t.astype(jnp.int32), axis=-1) // EGG_DIV
        ).astype(jnp.float32) / LOGIT_SCALE
        return jnp.mean(lse - t_logit)

    def __call__(self, tokens: jax.Array) -> jax.Array:
        """Full ``[B, T, V]`` logits — avoid in training (OOM on large V)."""
        embed_factors = self._prepare_embed_factors()
        h = self.encode(tokens, embed_factors=embed_factors)
        v = self.vocab_size
        # Prefer last_logits / chunked_nll; this path is for tiny-V debugging only.
        e = self._embed_rows(jnp.arange(v, dtype=jnp.int32), embed_factors)
        logits_i32 = int_matmul(h, e.T) // EGG_DIV
        return logits_i32.astype(jnp.float32) / LOGIT_SCALE


def _collect_lut_tables(params: dict) -> list[jax.Array]:
    """Walk the param tree for IntLUT ``table`` leaves."""
    tables: list[jax.Array] = []

    def _walk(node):
        if isinstance(node, dict):
            if "table" in node and isinstance(node["table"], jax.Array):
                tables.append(node["table"])
            for v in node.values():
                _walk(v)

    _walk(params)
    return tables


def lut_stats(model: IntRnnLM) -> tuple[int, float]:
    """How far LUT tables have moved from identity (changed entries, mean |Δ|)."""
    tables = _collect_lut_tables(params_pure_dict(model))
    if not tables:
        return 0, 0.0
    changed = 0
    total_abs = 0.0
    for table in tables:
        delta = jnp.abs(table.astype(jnp.int32) - _IDENTITY_LUT.astype(jnp.int32))
        changed += int(jnp.sum(delta > 0))
        total_abs += float(jnp.mean(delta))
    return changed, total_abs / len(tables)

# ── MiniPile tokenization → packed next-token batches ────────────────────────


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


# Set in main(); loss closes over it so ES never builds [B,T,V] logits.
LOGIT_CHUNK = 4096
CE_PARALLEL = 1


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


def count_params(model: nnx.Module) -> int:
    """Count ``nnx.Param`` leaves (ES-trainable, including tied embed)."""
    leaves = jax.tree.leaves(nnx.state(model, nnx.Param).to_pure_dict())
    return int(sum(v.size for v in leaves))


def save_checkpoint(
    model: IntRnnLM,
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


def load_checkpoint(model: IntRnnLM, path: str | Path) -> dict:
    """Load a trusted checkpoint created by :func:`save_checkpoint`."""
    payload = read_checkpoint(path)
    params = jax.tree.map(jnp.asarray, payload["params"])
    update_params(model, params)
    disable_candidates(model)
    return payload


def _all_integer_params(model: nnx.Module) -> bool:
    return all(
        jnp.issubdtype(v.dtype, jnp.integer)
        for v in jax.tree.leaves(params_pure_dict(model))
    )


DEFAULT_PROMPT = "Once upon a time"


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
    ctx_len = int(SEQ_LEN if max_ctx is None else max_ctx)
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


def main():
    global LOGIT_CHUNK, DELTA_CHUNK, CE_PARALLEL, GDN_FEAT_TILE

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
    parser.add_argument(
        "--update-alpha",
        type=float,
        default=0.12,
        help="Fraction of int params eligible for ±1 bins (lower = gentler LUT)",
    )
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
        help="Deprecated compatibility option; dense WY no longer feature-tiles.",
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

    if args.population % 2 != 0:
        raise SystemExit("--population must be even (integer_es antithetical pairs)")
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
    LOGIT_CHUNK = int(args.logit_chunk)
    DELTA_CHUNK = int(args.delta_chunk)
    CE_PARALLEL = max(1, int(args.ce_parallel))
    GDN_FEAT_TILE = max(1, int(args.gdn_feat_tile))
    chunk = None if args.candidate_chunk == 0 else args.candidate_chunk

    print(f"JAX devices: {jax.devices()}")
    print(f"Default backend: {jax.default_backend()}")
    print(
        f"GPU path: fused D→6D int8 in-proj + dense factorized WY; "
        f"widen --dim / --candidate-chunk / --batch; "
        f"CE logit_chunk={LOGIT_CHUNK} ce_parallel={CE_PARALLEL}; "
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
        f"Model: {NUM_LAYERS}L multi-head Gated Delta Rule-2 "
        f"[{args.delta_impl} C={DELTA_CHUNK}] (IntLinearLUT) + int8 MLP | "
        f"d={EMBED_DIM} heads={NUM_HEADS} hd={HEAD_DIM} ffn={FFN_DIM} "
        f"seq={SEQ_LEN} vocab={vocab_size:,} (tied learnable nnx.Embed)",
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
        seq_len=SEQ_LEN,
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
    assert _all_integer_params(model), "weights must stay integer after surgery"
    n_lut, m_lut = lut_stats(model)
    print(
        f"  params={n_params:,}  update={update_mode}  "
        f"LUT identity Δ={n_lut} mean|Δ|={m_lut:.3f}",
        flush=True,
    )

    run_config = {
        "dim": EMBED_DIM,
        "heads": NUM_HEADS,
        "head_dim": HEAD_DIM,
        "layers": NUM_LAYERS,
        "ffn_mult": FFN_MULT,
        "ffn_dim": FFN_DIM,
        "seq_len": SEQ_LEN,
        "batch": args.batch,
        "population": args.population,
        "rank": args.rank,
        "sigma_shift": args.sigma_shift,
        "candidate_chunk": chunk,
        "ce_parallel": CE_PARALLEL,
        "logit_chunk": LOGIT_CHUNK,
        "delta_impl": args.delta_impl,
        "delta_chunk": DELTA_CHUNK,
        "gdn_feat_tile": GDN_FEAT_TILE,
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

    wb = None
    if args.wandb:
        try:
            import wandb
        except ImportError as exc:
            raise SystemExit(
                "`--wandb` requires the wandb package.\n"
                "  uv pip install wandb"
            ) from exc
        run_name = args.wandb_run_name or (
            f"gdn2-d{EMBED_DIM}-L{NUM_LAYERS}-H{NUM_HEADS}-"
            f"B{args.batch}-P{args.population}-r{args.rank}"
        )
        wb = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config=run_config,
        )
        print(f"W&B run: {wb.url}", flush=True)

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
                    tokens_per_s = (args.batch * SEQ_LEN) / max(dt, 1e-6)
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

    assert _all_integer_params(model)
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
