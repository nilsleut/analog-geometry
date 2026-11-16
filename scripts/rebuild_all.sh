#!/usr/bin/env bash
# Rebuild every summary CSV and figure from the raw CSVs. Run INSIDE WSL from the repo root: bash scripts/rebuild_all.sh
set -eu
export AG_DATA_DIR=~/data/analog-geometry PYTHONPATH=src
P=~/.venvs/analog-geometry/bin/python
for ds in mnist cifar10; do $P -m analog_geometry.analyze --dataset "$ds"; done
$P -m analog_geometry.report_figures
