#!/usr/bin/env bash
# Thin wrapper: four-GPU character/byte-level GDN run (see run_h100_multigpu.sh).
exec env PROFILE=char bash "$(dirname "$0")/run_h100_multigpu.sh" "$@"
