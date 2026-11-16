"""Fairness check, levels (2026-11-16): quantization error per weight, (A) vs (C), measured on the programmed tiles.

Setting = levels sweep (no noise, no drift, t = 0). Per tile: e = alpha*w_programmed - w_digital, RMS in units of absmax,
plus the fraction of weights mapped exactly to 0 (|w_q| < 1e-9*absmax). Seeds 0-4 (noise-free -> no rep dependence).
Writes results/fairness_levels_error.csv (pooled over all weights of the network, per seed).
Usage: PYTHONPATH=src python scripts/fairness_levels_error.py
"""
from __future__ import annotations

import math
import warnings
from dataclasses import replace

import pandas as pd
import torch

from analog_geometry.analog import analog_model
from analog_geometry.common import RESULTS, SEEDS
from analog_geometry.sweep import ANCHOR, load_digital

warnings.filterwarnings("ignore")
torch.set_num_threads(12)
OFF = dict(prog_scale=0.0, drift_scale=0.0, read_scale=0.0, t_inference=0.0)
GRID = {"single_device": [8, 13, 15, 16, 27, 31, 32], "single_pair": [8, 16]}


def main():
    rows = []
    for ds in ("mnist", "cifar10"):
        for seed in SEEDS:
            dig, _ = load_digital(ds, seed)
            for conv, ks in GRID.items():
                for k in ks:
                    m = analog_model(dig, replace(ANCHOR, levels=k, **OFF), noise_seed=1000 * seed, converter=conv)
                    sse = n = nz = 0.0
                    for _, layer in m.named_analog_layers():
                        for tile in layer.analog_tiles():
                            w0 = tile.reference_combined_weights.detach().double()
                            wq = torch.as_tensor(tile.tile.get_weights()).double()
                            alpha = float(tile.alpha.reshape(-1)[0])
                            am = float(w0.abs().max())
                            sse += float((((alpha * wq - w0) / am) ** 2).sum())
                            n += w0.numel()
                            nz += float(((alpha * wq).abs() < 1e-9 * am).sum())
                    rows.append({"dataset": ds, "seed": seed, "converter": conv, "levels": k,
                                 "pooled_rms_err_rel": math.sqrt(sse / n), "frac_exact_zero": nz / n})
            print(ds, seed, "done", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(RESULTS / "fairness_levels_error.csv", index=False)
    print(d.groupby(["dataset", "converter", "levels"])[["pooled_rms_err_rel", "frac_exact_zero"]].mean().to_string())


if __name__ == "__main__":
    main()
