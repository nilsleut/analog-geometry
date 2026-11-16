#!/usr/bin/env bash
# Variant (C) differential pair: full noise + levels grid, overnight (decision 2026-11-16). Run from Git Bash on the host:
#   bash scripts/run_diffpair_grid.sh
# Same logic as run_cifar_rest.sh: one sweep per step, then analyze, then a LOCAL commit (no remote, no push).
# Differences: commits only the diffpair files (git commit -- <paths>), so parallel work in the tree is not swept in;
# an analyze failure is logged but does not block the commit of the raw CSV (figures are rebuilt from the CSVs later).
set -u
export MSYS_NO_PATHCONV=1
REPO_WIN="$(cd "$(dirname "$0")/.." && pwd)"
REPO_WSL="/mnt/c${REPO_WIN#/c}"
LOG="logs/diffpair_grid.log"
cd "$REPO_WIN" || exit 1
mkdir -p logs

py() {
  wsl -d Ubuntu-24.04 --cd "$REPO_WSL" -- bash -c \
    "export AG_DATA_DIR=~/data/analog-geometry PYTHONPATH=src; ~/.venvs/analog-geometry/bin/python $*"
}

step() {  # step <dataset> <label> <sweep args...>
  local ds="$1" label="$2"; shift 2
  echo "=== $(date '+%F %T') START $ds $label :: $*" >> "$LOG"
  if ! py -m analog_geometry.sweep --dataset "$ds" --only sweep --converter single_pair "$@" >> "$LOG" 2>&1; then
    echo "=== $(date '+%F %T') FAILED $ds $label -- stopping, no commit" >> "$LOG"; exit 1
  fi
  py -m analog_geometry.analyze --dataset "$ds" >> "$LOG" 2>&1 \
    || echo "=== $(date '+%F %T') analyze failed after $ds $label (raw CSV still committed)" >> "$LOG"
  local paths=()
  for f in "results/diffpair_$ds.csv" "results/findings_diffpair_$ds.csv" "results/sweep_summary_diffpair_$ds.csv"; do
    [ -e "$f" ] && paths+=("$f")
  done
  git add -- "${paths[@]}" >> "$LOG" 2>&1
  for try in 1 2 3; do  # retry if the index is locked by a parallel git call
    if git commit -q -m "Variante C (Differenzpaar) $ds: $label" \
         -- "${paths[@]}" >> "$LOG" 2>&1; then break; fi
    sleep 10
  done
  echo "=== $(date '+%F %T') DONE $ds $label ($(git log --oneline -1))" >> "$LOG"
}

for ds in mnist cifar10; do
  step "$ds" "Rauschskalierung 0-64" --sweeps prog_noise
  step "$ds" "Leitwertstufen 8/16/32/64/inf" --sweeps levels
done
echo "=== $(date '+%F %T') ALL DONE" >> "$LOG"
