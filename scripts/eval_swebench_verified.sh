#!/usr/bin/env bash
set -euo pipefail

PREDICTIONS_PATH="${1:?Usage: $0 /path/to/preds.json [run_id]}"
RUN_ID="${2:-$(basename "$(dirname "${PREDICTIONS_PATH}")")}" 
MAX_WORKERS="${MAX_WORKERS:-8}"

python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-Bench_Verified \
  --split test \
  --predictions_path "${PREDICTIONS_PATH}" \
  --max_workers "${MAX_WORKERS}" \
  --run_id "${RUN_ID}"
