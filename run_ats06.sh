#!/usr/bin/env bash
set -euo pipefail

# Put this file at the perspective-llm repository root.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

mkdir -p logs
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="logs/ats06_${STAMP}.log"

export PYTHONUNBUFFERED=1

echo "[ATS06] repo: $REPO_ROOT"
echo "[ATS06] log:  $LOG"
echo "[ATS06] args: $*"

python -u ATS/models/ats_06_cear_g_ca_terminal.py "$@" 2>&1 | tee "$LOG"
