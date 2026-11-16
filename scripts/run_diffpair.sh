#!/usr/bin/env bash
# Gate 2026-11-16: CMO drift with differential-pair converter (single_pair), drift 1 d / 1 wk / 1 yr, both datasets.
# Run INSIDE WSL from the repo root:  bash scripts/run_diffpair.sh
set -eu
export AG_DATA_DIR=~/data/analog-geometry PYTHONPATH=src
P=~/.venvs/analog-geometry/bin/python
mkdir -p logs
for ds in mnist cifar10; do
  $P -m analog_geometry.sweep --dataset "$ds" --only sweep --sweeps drift_time --times 86400 604800 31536000 \
     --converter single_pair > "logs/diffpair_$ds.log" 2>&1
  $P -m analog_geometry.analyze --dataset "$ds" >> "logs/diffpair_$ds.log" 2>&1
done
echo ALL OK
grep -h "RATE\|wall" logs/diffpair_*.log
