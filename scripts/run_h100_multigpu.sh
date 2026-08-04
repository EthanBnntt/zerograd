#!/usr/bin/env bash
# Four-GPU loss-only distributed integer GDN run — set PROFILE=char|qwen.
#
#   PROFILE=char bash scripts/run_h100_multigpu.sh   # diagnostic byte/char run
#   PROFILE=qwen bash scripts/run_h100_multigpu.sh   # ~104M-param Qwen run
#
# ``scripts/run_h100_char_multigpu.sh`` and ``scripts/run_h100_qwen_multigpu.sh``
# are 2-line wrappers around this script with PROFILE pre-set, kept for
# backwards compatibility.
set -euo pipefail
cd "$(dirname "$0")/.."

PROFILE="${PROFILE:-char}"
case "$PROFILE" in
  char|qwen) ;;
  *) echo "PROFILE must be 'char' or 'qwen' (got '$PROFILE')" >&2; exit 1 ;;
esac

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

# Per-profile defaults; any of these can still be overridden by exporting the
# corresponding env var (e.g. ``STEPS=100 PROFILE=char bash ...``).
if [ "$PROFILE" = "char" ]; then
  _STEPS=500
  _MAX_HOURS=0
  _DIM=1024
  _HEADS=8
  _LAYERS=4
  _BATCH=32
  _LOGIT_CHUNK=512
  _TOKENIZER_MODE=byte
  _CHECKPOINT_OUT="checkpoints/int-gdn-char-4gpu-latest.pkl"
  _CHECKPOINT_EVERY=50
  _LOG_EVERY=5
  _WANDB_RUN_NAME="h100-4x-gdn2-char-d1024-L4-pop256"
else
  _STEPS=1000000
  _MAX_HOURS=4
  _DIM=384
  _HEADS=6
  _LAYERS=4
  _BATCH=16
  _LOGIT_CHUNK=8192
  _TOKENIZER_MODE=qwen
  _CHECKPOINT_OUT="checkpoints/int-gdn-qwen-4gpu-latest.pkl"
  _CHECKPOINT_EVERY=1000
  _LOG_EVERY=50
  _WANDB_RUN_NAME="h100-4x-gdn2-qwen-d384-L4-pop256"
fi

CHECKPOINT_OUT="${CHECKPOINT_OUT:-$_CHECKPOINT_OUT}"
RESUME_ARGS=()
if [ "$PROFILE" = "qwen" ] && [ -f "${RESUME_FROM:-$CHECKPOINT_OUT}" ]; then
  RESUME_ARGS+=(--resume-from "${RESUME_FROM:-$CHECKPOINT_OUT}")
fi

MAX_HOURS_ARGS=()
if [ "${MAX_HOURS:-$_MAX_HOURS}" != "0" ]; then
  MAX_HOURS_ARGS+=(--max-hours "${MAX_HOURS:-$_MAX_HOURS}")
fi

exec $PYTHON examples/train_int_rnn_minipile_multigpu.py \
  --steps "${STEPS:-$_STEPS}" \
  --devices "${DEVICES:-4}" \
  --dim "${DIM:-$_DIM}" \
  --heads "${HEADS:-$_HEADS}" \
  --layers "${LAYERS:-$_LAYERS}" \
  --ffn-mult "${FFN_MULT:-4}" \
  --seq-len "${SEQ_LEN:-512}" \
  --batch "${BATCH:-$_BATCH}" \
  --population "${POP:-256}" \
  --rank "${RANK:-8}" \
  --sigma-shift "${SIGMA_SHIFT:-4}" \
  --update-alpha "${UPDATE_ALPHA:-0.02}" \
  --alpha-decay "${ALPHA_DECAY:-0.001}" \
  --candidate-chunk "${CAND_CHUNK:-32}" \
  --logit-chunk "${LOGIT_CHUNK:-$_LOGIT_CHUNK}" \
  --tokenizer-mode "${TOKENIZER_MODE:-$_TOKENIZER_MODE}" \
  --checkpoint-out "$CHECKPOINT_OUT" \
  --checkpoint-every "${CHECKPOINT_EVERY:-$_CHECKPOINT_EVERY}" \
  --log-every "${LOG_EVERY:-$_LOG_EVERY}" \
  --wandb-project "${WANDB_PROJECT:-zerograd-int-gdn}" \
  --wandb-run-name "${WANDB_RUN_NAME:-$_WANDB_RUN_NAME}" \
  "${MAX_HOURS_ARGS[@]}" \
  "${RESUME_ARGS[@]}" \
  "$@"
