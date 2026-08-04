#!/usr/bin/env bash
# Four-GPU loss-only Qwen-tokenized ~100M-parameter integer GDN run.
set -euo pipefail
cd "$(dirname "$0")/.."

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-true}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$HOME/.cache/jax/zerograd-h100-4gpu}"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="${JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS:-1}"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_enable_triton_gemm=true}"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

PYTHON=.venv/bin/python
if [ ! -x "$PYTHON" ]; then
  PYTHON="uv run python"
fi

exec $PYTHON examples/train_int_rnn_minipile_multigpu.py \
  --steps "${STEPS:-300}" \
  --devices "${DEVICES:-4}" \
  --dim "${DIM:-384}" \
  --heads "${HEADS:-6}" \
  --layers "${LAYERS:-4}" \
  --ffn-mult "${FFN_MULT:-4}" \
  --seq-len "${SEQ_LEN:-512}" \
  --batch "${BATCH:-16}" \
  --population "${POP:-256}" \
  --rank "${RANK:-8}" \
  --sigma-shift "${SIGMA_SHIFT:-4}" \
  --update-alpha "${UPDATE_ALPHA:-0.02}" \
  --alpha-decay "${ALPHA_DECAY:-0.001}" \
  --candidate-chunk "${CAND_CHUNK:-32}" \
  --logit-chunk "${LOGIT_CHUNK:-8192}" \
  --tokenizer-mode qwen \
  --checkpoint-out "${CHECKPOINT_OUT:-checkpoints/int-gdn-qwen-4gpu-latest.pkl}" \
  --checkpoint-every "${CHECKPOINT_EVERY:-25}" \
  --log-every "${LOG_EVERY:-5}" \
  --wandb-project "${WANDB_PROJECT:-zerograd-int-gdn}" \
  --wandb-run-name "${WANDB_RUN_NAME:-h100-4x-gdn2-qwen-d384-L4-pop256}" \
  "$@"
