"""Pure-int8 Gated DeltaNet-2 LM architecture (model half of ``train_int_rnn_minipile.py``).

Architecture (all int8 until the head). The main block is ``IntLinearLUT``:

  int8 @ int8 → int32 accum → int8 requant → learnable IntLUT

  Qwen3.6 subword ids → int8 ``IntEmbedding`` (surgery → ``ZgIntEmbedding``)
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

See ``examples/train_int_rnn_minipile.py`` for the CLI entry point, and the
sibling modules ``data.py`` (tokenizer/dataset/batching), ``checkpoint.py``
(save/load), and ``train_loop.py`` (eval/sampling/logging) for the rest of
the training program.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from flax import nnx

from zerograd import (
    IntAffine,
    IntEmbedding,
    IntLinear,
    IntLinearLUT,
    IntLUT,
    egg_clip_cast,
)
from zerograd._integer import egg_matmul_divisor, int_matmul
from zerograd._nnx import LayerIndex, params_pure_dict

# ── Architecture defaults (overridden by CLI via ``configure_architecture``) ─
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
        # IntEmbedding → surgery → ZgIntEmbedding (TABLE, row-sparse ES factors).
        self.embed = IntEmbedding(self.vocab_size, EMBED_DIM, rngs=rngs)
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


def count_params(model: nnx.Module) -> int:
    """Count ``nnx.Param`` leaves (ES-trainable, including tied embed)."""
    leaves = jax.tree.leaves(nnx.state(model, nnx.Param).to_pure_dict())
    return int(sum(v.size for v in leaves))


def all_integer_params(model: nnx.Module) -> bool:
    """Whether every ``nnx.Param`` leaf still has an integer dtype."""
    return all(
        jnp.issubdtype(v.dtype, jnp.integer)
        for v in jax.tree.leaves(params_pure_dict(model))
    )
