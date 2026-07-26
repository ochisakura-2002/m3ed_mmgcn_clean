#!/usr/bin/env bash
set -euo pipefail

# Config preparation and command generation only. This script never executes
# the generated training commands.
# The paired primary matrix expands to 32 runs (16 full + 16 causal); the
# disabled three-seed matrix expands the same 16 pair keys to 96 runs.
# MERC_LONG_TRAINING_RESOLVED_ROOT may redirect config-only materialization to
# a temporary directory for regression testing.

MATRIX_PATH="${1:-configs/benchmarks/long_training/iemocap_clean/primary_seed42.yaml}"
MODE="${2:-check}"
EXPERIMENT_DATE="${MERC_EXPERIMENT_DATE:-$(date +%Y%m%d)}"
BATCH_ID="${MERC_LONG_TRAINING_BATCH_ID:-formal_long32_$(date +%Y%m%d_%H%M%S)}"
RESOLVED_ROOT_ARGS=()

if [[ "${MODE}" != "check" && "${MODE}" != "prepare" ]]; then
  echo "Usage: $0 [matrix.yaml] [check|prepare]" >&2
  exit 2
fi

if [[ -n "${MERC_LONG_TRAINING_RESOLVED_ROOT:-}" ]]; then
  RESOLVED_ROOT_ARGS=(--resolved-root "${MERC_LONG_TRAINING_RESOLVED_ROOT}")
fi

python -u scripts/workflows/benchmarks/prepare_long_training.py \
  "${MATRIX_PATH}" \
  "${MODE}" \
  "${EXPERIMENT_DATE}" \
  --batch-id "${BATCH_ID}" \
  "${RESOLVED_ROOT_ARGS[@]}"
