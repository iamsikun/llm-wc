#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-ollama/gemma3:4b}"
WORKERS="${WORKERS:-4}"
OUTPUT_BASE="${OUTPUT_BASE:-runs/swebench_verified}"

SANITIZED_MODEL="${MODEL//\//_}"
SANITIZED_MODEL="${SANITIZED_MODEL//:/_}"
OUTPUT_DIR="${2:-${OUTPUT_BASE}/${SANITIZED_MODEL}}"

mkdir -p "${OUTPUT_DIR}"

mini-extra swebench \
  --config config/swebench_ollama.yaml \
  --model "${MODEL}" \
  --subset verified \
  --split test \
  --workers "${WORKERS}" \
  --output "${OUTPUT_DIR}"
