#!/usr/bin/env bash
set -euo pipefail

# ATS 06 replication run
# - seed = 0
# - 280 epochs exactly (early stopping disabled)
# - model checkpoint every epoch
# - crash-safe training_history.csv updated every epoch
# - TensorBoard scalars enabled by default

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

SCRIPT="ATS/models/ats_06_cear_g_ca.py"
if [[ ! -f "$SCRIPT" ]]; then
    echo "[ATS06] ERROR: missing $REPO_ROOT/$SCRIPT"
    exit 1
fi

# Refuse to run an older non-replication script by accident.
if ! grep -q -- '"--checkpoint-every"' "$SCRIPT"; then
    echo "[ATS06] ERROR: $SCRIPT is not the updated replication version."
    echo "Replace it with ats_06_cear_g_ca_replication.py first."
    exit 1
fi

mkdir -p logs
mkdir -p results/ATS/cear_g_ca

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="rep_seed0_${STAMP}"
RESULT_DIR="results/ATS/cear_g_ca/${RUN_NAME}"
LOG="logs/ats06_${RUN_NAME}.log"

export PYTHONUNBUFFERED=1

echo
printf '%s\n' "======================================================================"
printf '%s\n' "ATS 06 — REPLICATION RUN"
printf '%s\n' "======================================================================"
echo "repo:        $REPO_ROOT"
echo "script:      $SCRIPT"
echo "results:     $RESULT_DIR"
echo "log:         $LOG"
echo "seed:        0"
echo "epochs:      280"
echo "early stop:  disabled"
echo "checkpoint:  every epoch"
echo "TensorBoard: $RESULT_DIR/tensorboard"
echo "extra args:  $*"
printf '%s\n' "======================================================================"
echo

python -u "$SCRIPT" \
    --device cuda \
    --seed 0 \
    --epochs 280 \
    --patience 0 \
    --checkpoint-every 1 \
    --result-dir "$RESULT_DIR" \
    "$@" \
    2>&1 | tee "$LOG"

echo
printf '%s\n' "======================================================================"
echo "ATS06 replication finished"
echo "results: $RESULT_DIR"
echo "log:     $LOG"
echo

echo "Per-epoch weights:"
echo "  $RESULT_DIR/checkpoints/epoch_001.pt ... epoch_280.pt"
echo "Best validation checkpoint:"
echo "  $RESULT_DIR/best.pt"
echo "Latest full resumable checkpoint:"
echo "  $RESULT_DIR/latest_full.pt"
echo "TensorBoard events:"
echo "  $RESULT_DIR/tensorboard/"
printf '%s\n' "======================================================================"
