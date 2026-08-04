#!/usr/bin/env bash
# Launch MiniPile int-GDN training with W&B on the H100 box.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

export PATH="$HOME/.local/bin:$PATH"
export WANDB_API_KEY="${WANDB_API_KEY:-$(python3 -c 'import netrc; print(netrc.netrc().authenticators("api.wandb.ai")[2])')}"
export WANDB="${WANDB:-1}"
export WANDB_PROJECT="${WANDB_PROJECT:-zerograd-int-gdn}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-h100-gdn2-d2048-L12-opt}"
export STEPS="${STEPS:-2000}"
export LOG_EVERY="${LOG_EVERY:-5}"
export GEN_EVERY="${GEN_EVERY:-0}"

LOG="logs/train_wandb_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to $LOG"
echo "WANDB_PROJECT=$WANDB_PROJECT RUN=$WANDB_RUN_NAME STEPS=$STEPS key_len=${#WANDB_API_KEY}"
exec bash scripts/run_h100_minipile.sh "$@" 2>&1 | tee "$LOG"
