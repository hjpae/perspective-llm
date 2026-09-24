#!/usr/bin/env bash
set -euo pipefail

# Run from the perspective-llm repo root.
REPO_ROOT="$(pwd)"

# Prefer the most recent seed-0 ATS06 replication unless ATS06_DIR is supplied.
if [[ -z "${ATS06_DIR:-}" ]]; then
  ATS06_DIR="$(ls -dt results/ATS/cear_g_ca/rep_seed0_* 2>/dev/null | head -n 1 || true)"
fi

if [[ -z "${ATS06_DIR}" || ! -d "${ATS06_DIR}" ]]; then
  echo "Could not find ATS06 seed-0 result directory."
  echo "Set it explicitly, e.g.:"
  echo "  ATS06_DIR=results/ATS/cear_g_ca/rep_seed0_20260924_024922 ./run_ats07.sh"
  exit 1
fi

SCRIPT="ATS/models/ats_07_strong_baselines_ca.py"
if [[ ! -f "$SCRIPT" ]]; then
  echo "Missing $SCRIPT"
  exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
RESULT_DIR="results/ATS/strong_baselines_ca/seed0_${STAMP}"
LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/ats07_seed0_${STAMP}.log"
mkdir -p "$RESULT_DIR" "$LOG_DIR"

echo "ATS06_DIR:  $ATS06_DIR"
echo "RESULT_DIR: $RESULT_DIR"
echo "LOG_FILE:   $LOG_FILE"

python -u "$SCRIPT" \
  --ats06-dir "$ATS06_DIR" \
  --result-dir "$RESULT_DIR" \
  --device cuda \
  --seed 0 \
  --epochs 120 \
  --patience 20 \
  --min-delta 1e-4 \
  --retrieval-k 3,5,10 \
  --retrieval-temperature 0.20 \
  --val-ats-permutations 3 \
  --test-ats-permutations 12 \
  --bootstrap 3000 \
  "$@" 2>&1 | tee "$LOG_FILE"

echo
echo "======================================================================"
echo "ATS07 finished"
echo "results: $RESULT_DIR"
echo "log:     $LOG_FILE"
echo "======================================================================"
