#!/usr/bin/env bash
# H100-oriented MiniPile trainer (CUDA JAX). Run from repo root on the instance.
#
# Tuned for INT8 Tensor Core occupancy within ~80GB:
#   - fused D→6D mixer in-proj + wide body (--dim)
#   - memory-light WY (no [C,C,hd] ratios)
#   - modest --ce-parallel (float CE is VRAM-heavy); prefer --candidate-chunk
#     so more int8 forwards run concurrently
#   - --rank 4 fattens ES factor GEMMs vs default rank-2
set -euo pipefail
cd "$(dirname "$0")/.."

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-true}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"
# Prefer large matmul autotune / less host sync chatter.
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_enable_triton_gemm=true}"

# Use .venv python when available (H100 setup), else uv run.
if [ -x .venv/bin/python ]; then
  PYTHON=.venv/bin/python
else
  PYTHON="uv run python"
fi

$PYTHON examples/train_int_rnn_minipile.py \
  --steps "${STEPS:-2000}" \
  --dim "${DIM:-2048}" \
  --heads "${HEADS:-16}" \
  --layers "${LAYERS:-12}" \
  --ffn-mult "${FFN_MULT:-4}" \
  --seq-len "${SEQ_LEN:-512}" \
  --batch "${BATCH:-16}" \
  --population "${POP:-32}" \
  --rank "${RANK:-4}" \
  --candidate-chunk "${CAND_CHUNK:-12}" \
  --ce-parallel "${CE_PARALLEL:-2}" \
  --delta-impl chunkwise \
  --delta-chunk "${DELTA_CHUNK:-64}" \
  --gdn-feat-tile "${GDN_FEAT_TILE:-32}" \
  --logit-chunk "${LOGIT_CHUNK:-12288}" \
  --prefetch 8 \
  --log-every 10 \
  --gen-every 0 \
  "$@"
