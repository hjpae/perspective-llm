#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

ATS06_DIR="${ATS06_DIR:-$(ls -dt results/ATS/cear_g_ca/rep_seed0_* 2>/dev/null | head -n 1)}"
if [[ -z "${ATS06_DIR}" ]]; then
  echo "ERROR: no results/ATS/cear_g_ca/rep_seed0_* found"
  exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
RESULT_DIR="results/ATS/personamem_stress/corrected_seed0_${STAMP}"
LOG_FILE="logs/ats08b_pm_corrected_${STAMP}.log"
mkdir -p "${RESULT_DIR}" logs

echo "ATS06_DIR:  ${ATS06_DIR}"
echo "RESULT_DIR: ${RESULT_DIR}"
echo "LOG_FILE:   ${LOG_FILE}"

python -u ATS/models/ats_08b_personamem_corrected_geometry.py \
  --ats06-dir "${ATS06_DIR}" \
  --ats06-source ATS/models/ats_06_cear_g_ca.py \
  --personamem-root data/personamem_v2 \
  --result-dir "${RESULT_DIR}" \
  --device cuda \
  --seed 0 \
  --pre-old 6 \
  --stress-steps 8 \
  --recovery-old 4 \
  --bootstrap 3000 \
  2>&1 | tee "${LOG_FILE}"
