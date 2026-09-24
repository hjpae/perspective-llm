#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
ATS06_DIR="${ATS06_DIR:-$(ls -dt results/ATS/cear_g_ca/rep_seed0_* | head -n 1)}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RESULT_DIR="results/ATS/personamem_adaptation/seed0_${STAMP}"
LOG_FILE="logs/ats08c_personamem_adaptation_${STAMP}.log"
mkdir -p "$RESULT_DIR" logs

echo "ATS06_DIR:  $ATS06_DIR"
echo "RESULT_DIR: $RESULT_DIR"
echo "LOG_FILE:   $LOG_FILE"

python -u ATS/models/ats_08c_personamem_adaptation.py \
  --ats06-dir "$ATS06_DIR" \
  --ats06-source ATS/models/ats_06_cear_g_ca.py \
  --ats08b-source ATS/models/ats_08b_personamem_corrected_geometry.py \
  --personamem-root data/personamem_v2 \
  --result-dir "$RESULT_DIR" \
  --device cuda \
  --seed 0 \
  --epochs 120 \
  --patience 20 \
  --lr 1e-4 \
  2>&1 | tee "$LOG_FILE"
