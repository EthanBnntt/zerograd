#!/usr/bin/env bash
# Thin wrapper: four-GPU Qwen-tokenized ~100M-parameter GDN run (see run_h100_multigpu.sh).
exec env PROFILE=qwen bash "$(dirname "$0")/run_h100_multigpu.sh" "$@"
