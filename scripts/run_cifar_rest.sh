#!/usr/bin/env bash
# CIFAR-10 rest grid, overnight run (decision 2026-11-15). Run from Git Bash on the Windows host:
#   bash scripts/run_cifar_rest.sh
# Computation runs in WSL (venv ~/.venvs/analog-geometry); git commits are local only (no remote, no push).
# Order (user): prog 1,2,8 -> drift series incl. offset controls (2a; 2b = cka_stimc, + pre-activations)
#               -> levels -> prog 32 + periphery_default -> remaining full-grid points (ideal, wan2022).
# Stops at the first failing step; nothing is committed for a failed step.
set -u
export MSYS_NO_PATHCONV=1
REPO_WIN="$(cd "$(dirname "$0")/.." && pwd)"
REPO_WSL="/mnt/c${REPO_WIN#/c}"
LOG="logs/sweep_cifar10_rest.log"
cd "$REPO_WIN" || exit 1
mkdir -p logs

py() {  # run a module inside WSL with the project venv
  wsl -d Ubuntu-24.04 --cd "$REPO_WSL" -- bash -c \
    "export AG_DATA_DIR=~/data/analog-geometry PYTHONPATH=src; ~/.venvs/analog-geometry/bin/python $*"
}

step() {  # step <label> <sweep args...>
  local label="$1"; shift
  echo "=== $(date '+%F %T') START $label :: $*" >> "$LOG"
  if ! py -m analog_geometry.sweep --dataset cifar10 --only sweep "$@" >> "$LOG" 2>&1; then
    echo "=== $(date '+%F %T') FAILED $label -- stopping, no commit" >> "$LOG"; exit 1
  fi
  if ! py -m analog_geometry.analyze --dataset cifar10 >> "$LOG" 2>&1; then
    echo "=== $(date '+%F %T') ANALYZE FAILED after $label -- stopping, no commit" >> "$LOG"; exit 1
  fi
  git add results/ figures/ >> "$LOG" 2>&1
  git commit -q -m "CIFAR: $label" >> "$LOG" 2>&1 \
    || echo "(nothing to commit for $label)" >> "$LOG"
  echo "=== $(date '+%F %T') DONE $label ($(git log --oneline -1))" >> "$LOG"
}

step "Rauschskalierung 1/2/8"                --sweeps prog_noise --prog-scales 1 2 8
step "Drift-Reihe + Versatz-Kontrollen (2a, 2b, Prä-Aktivierungen)" --sweeps drift_time drift_no_mean --preact
step "Leitwertstufen 8/16/32/64/inf"         --sweeps levels
step "Rauschskalierung 32 + Default-Peripherie" --sweeps prog_noise periphery_default --prog-scales 32
step "Rest des vollen Rasters (ideal, Wan2022)" --sweeps ideal wan2022_time
echo "=== $(date '+%F %T') ALL DONE" >> "$LOG"
