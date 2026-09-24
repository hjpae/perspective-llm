#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# ATS 06 — long convergence run
#
# Defaults:
#   max epochs : 300
#   patience   : 40
#   min_delta  : 1e-5
#   device     : cuda
#
# Extra CLI arguments can still be appended, e.g.
#   ./run_ats06.sh --metric-rank 16
#
# ============================================================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# ------------------------------------------------------------
# Find the CLI-capable ATS06 script.
# Prefer the terminal version because the earlier non-CLI
# script silently ignored arguments such as --debug.
# ------------------------------------------------------------

if [[ -f "ATS/models/ats_06_cear_g_ca_terminal.py" ]]; then
    SOURCE_SCRIPT="ATS/models/ats_06_cear_g_ca_terminal.py"
elif [[ -f "ATS/models/ats_06_cear_g_ca.py" ]]; then
    SOURCE_SCRIPT="ATS/models/ats_06_cear_g_ca.py"
else
    echo "[ATS06] ERROR: could not find ATS06 Python script."
    exit 1
fi

# Make sure we actually selected the CLI version.
if ! grep -q -- '"--epochs"' "$SOURCE_SCRIPT"; then
    echo "[ATS06] ERROR:"
    echo "  $SOURCE_SCRIPT does not appear to be the CLI-capable version."
    echo "  It has no --epochs argument."
    echo
    echo "Use ats_06_cear_g_ca_terminal.py or replace"
    echo "ATS/models/ats_06_cear_g_ca.py with the terminal version."
    exit 1
fi


# ------------------------------------------------------------
# Run directories
# ------------------------------------------------------------

mkdir -p logs
mkdir -p .ats06_run

STAMP="$(date +%Y%m%d_%H%M%S)"

LOG="logs/ats06_long_${STAMP}.log"
RESULT_DIR="results/ATS/cear_g_ca/longrun_${STAMP}"

# Temporary copy: original Python source remains untouched.
RUN_SCRIPT=".ats06_run/ats_06_cear_g_ca_${STAMP}.py"
cp "$SOURCE_SCRIPT" "$RUN_SCRIPT"


# ------------------------------------------------------------
# Patch convergence settings in TEMPORARY COPY ONLY.
#
# Current ATS06 Python script has:
#   PATIENCE = 5
#   best_val - 1e-4
#
# For this run:
#   PATIENCE = 40
#   min_delta = 1e-5
# ------------------------------------------------------------

python - "$RUN_SCRIPT" <<'PY'
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")

# Patience
text, n_patience = re.subn(
    r"(?m)^PATIENCE\s*=\s*\d+\s*$",
    "PATIENCE = 40",
    text,
    count=1,
)

if n_patience != 1:
    raise RuntimeError(
        "Could not locate exactly one PATIENCE constant in ATS06 script."
    )

# Validation improvement threshold.
# Accept either the original 1e-4 or an already-patched 1e-5.
if "best_val - 1e-4" in text:
    text = text.replace(
        "best_val - 1e-4",
        "best_val - 1e-5",
        1,
    )
elif "best_val - 1e-5" not in text:
    raise RuntimeError(
        "Could not locate the validation min-delta expression."
    )

path.write_text(text, encoding="utf-8")

print("[ATS06] temporary training script patched:")
print("        PATIENCE = 40")
print("        min_delta = 1e-5")
PY


# ------------------------------------------------------------
# Environment / logging
# ------------------------------------------------------------

export PYTHONUNBUFFERED=1

echo
echo "======================================================================"
echo "ATS 06 — LONG CONVERGENCE RUN"
echo "======================================================================"
echo "[ATS06] repo:       $REPO_ROOT"
echo "[ATS06] source:     $SOURCE_SCRIPT"
echo "[ATS06] run script: $RUN_SCRIPT"
echo "[ATS06] results:    $RESULT_DIR"
echo "[ATS06] log:        $LOG"
echo
echo "[ATS06] max epochs: 300"
echo "[ATS06] patience:   40"
echo "[ATS06] min delta:  1e-5"
echo "[ATS06] device:     cuda"
echo "[ATS06] extra args: $*"
echo "======================================================================"
echo


# ------------------------------------------------------------
# Training
#
# Defaults come FIRST.
# Therefore anything supplied after ./run_ats06.sh can override
# them because argparse uses the last occurrence.
#
# Example:
#   ./run_ats06.sh --epochs 500
# ------------------------------------------------------------

python -u "$RUN_SCRIPT" \
    --device cuda \
    --epochs 300 \
    --result-dir "$RESULT_DIR" \
    "$@" \
    2>&1 | tee "$LOG"


echo
echo "======================================================================"
echo "ATS06 finished"
echo "results: $RESULT_DIR"
echo "log:     $LOG"
echo "======================================================================"