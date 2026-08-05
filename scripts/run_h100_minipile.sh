#!/usr/bin/env bash
# H100-oriented MiniPile trainer (CUDA JAX). Run from repo root on the instance.
#
# Tuned for INT8 Tensor Core occupancy within ~80GB:
#   - fused D→6D mixer in-proj + wide body (--dim)
#   - dense factorized WY (no [C,C,hd] ratios)
#   - population 256 (128 antithetical directions), evaluated in dense chunks
#     of 32 (~51GB measured peak per chunk)
#   - conservative local perturbations and 2% decaying bin updates
set -euo pipefail
cd "$(dirname "$0")/.."

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-true}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$HOME/.cache/jax/zerograd-h100}"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="${JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS:-1}"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"
# Prefer large matmul autotune / less host sync chatter.
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_enable_triton_gemm=true}"

# Use .venv python when available (H100 setup), else uv run.
if [ -x .venv/bin/python ]; then
  PYTHON=.venv/bin/python
else
  PYTHON="uv run python"
fi

WANDB_ARGS=()
if [ "${WANDB:-1}" != "0" ]; then
  WANDB_ARGS+=(--wandb)
  if [ -n "${WANDB_PROJECT:-}" ]; then
    WANDB_ARGS+=(--wandb-project "$WANDB_PROJECT")
  fi
  if [ -n "${WANDB_ENTITY:-}" ]; then
    WANDB_ARGS+=(--wandb-entity "$WANDB_ENTITY")
  fi
  if [ -n "${WANDB_RUN_NAME:-}" ]; then
    WANDB_ARGS+=(--wandb-run-name "$WANDB_RUN_NAME")
  fi
fi

CHECKPOINT_ARGS=()
if [ -n "${CHECKPOINT_OUT:-}" ]; then
  CHECKPOINT_ARGS+=(--checkpoint-out "$CHECKPOINT_OUT")
  CHECKPOINT_ARGS+=(--checkpoint-every "${CHECKPOINT_EVERY:-0}")
fi

$PYTHON examples/train_int_rnn_minipile.py \
  --steps "${STEPS:-2000}" \
  --dim "${DIM:-2048}" \
  --heads "${HEADS:-16}" \
  --layers "${LAYERS:-12}" \
  --ffn-mult "${FFN_MULT:-4}" \
  --seq-len "${SEQ_LEN:-512}" \
  --batch "${BATCH:-16}" \
  --population "${POP:-256}" \
  --rank "${RANK:-4}" \
  --sigma-shift "${SIGMA_SHIFT:-3}" \
  --update-alpha "${UPDATE_ALPHA:-0.02}" \
  --alpha-decay "${ALPHA_DECAY:-0.001}" \
  --candidate-chunk "${CAND_CHUNK:-32}" \
  --ce-parallel "${CE_PARALLEL:-2}" \
  --delta-impl chunkwise \
  --delta-chunk "${DELTA_CHUNK:-64}" \
  --gdn-feat-tile "${GDN_FEAT_TILE:-32}" \
  --logit-chunk "${LOGIT_CHUNK:-12288}" \
  --tokenizer-mode "${TOKENIZER_MODE:-qwen}" \
  --prefetch 8 \
  --log-every "${LOG_EVERY:-10}" \
  --gen-every "${GEN_EVERY:-0}" \
  "${WANDB_ARGS[@]}" \
  "${CHECKPOINT_ARGS[@]}" \
  "$@"
