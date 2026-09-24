#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Load repo-local .env into the shell environment.
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
  # Defensive cleanup for CRLF-formatted .env files
  export OPENAI_API_KEY="${OPENAI_API_KEY//$'\r'/}"
fi

if [[ -z "${ATS08C_DIR:-}" ]]; then
  for d in $(ls -dt results/ATS/personamem_adaptation/seed0_*); do
    if [[ -f "$d/experiment_metadata.json" && -f "$d/best.pt" ]]; then
      ATS08C_DIR="$d"
      break
    fi
  done
fi

MODEL="${MODEL:-gpt-5-mini}"
MAX_QUERIES="${MAX_QUERIES:-196}"
WORKERS="${WORKERS:-8}"
DEVICE="${DEVICE:-cuda}"

STAMP="$(date +%Y%m%d_%H%M%S)"

RESULT_DIR="${RESULT_DIR:-results/ATS/rag_history_budget/${MODEL}_${STAMP}}"
LOG_FILE="${LOG_FILE:-logs/ats09a_rag_${MODEL}_${STAMP}.log}"

mkdir -p "$RESULT_DIR" logs

: "${OPENAI_API_KEY:?OPENAI_API_KEY is not set}"

echo "ATS08C_DIR:  $ATS08C_DIR"
echo "MODEL:       $MODEL"
echo "MAX_QUERIES: $MAX_QUERIES"
echo "WORKERS:     $WORKERS"
echo "DEVICE:      $DEVICE"
echo "RESULT_DIR:  $RESULT_DIR"

python -u ATS/models/ats_09a_actual_rag_history_budget.py \
  --ats08c-dir "$ATS08C_DIR" \
  --personamem-root data/personamem_v2 \
  --result-dir "$RESULT_DIR" \
  --model "$MODEL" \
  --max-queries "$MAX_QUERIES" \
  --workers "$WORKERS" \
  --device "$DEVICE" \
  --seed 0 \
  2>&1 | tee "$LOG_FILE"
