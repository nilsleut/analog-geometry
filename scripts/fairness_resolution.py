"""Fairness check (C) vs (A) for noise and levels (2026-11-16): effective weight resolution per converter.

Analytic (from aihwkit 1.2.0 code, CMO model, acceptance 2 %):
  programming noise  sigma_prog(g) = c0 + c1*g  [uS], c0 = 0.0112185391, c1 = 0.01129027418  (conductance-DEPENDENT)
  floor at t = 1 s   relaxation sigma = std[1] = 0.41183342 uS (g-independent, NOT scaled by prog_scale)
                     read sigma = K*log10(g)*sqrt(ln((1 + t_read)/(2 t_read))), K = 0.0277316483, t_read = 1e-6
  (A) single device: g = g_min + (w - w_min)*(g_max - g_min)/(w_max - w_min); one device per weight
  (C) pair:          G+ = g_min + max(w,0)*(g_max - g_min)/absmax, G- = g_min + max(-w,0)*...; two devices per weight
  Levels: k levels per device in [g_min, g_max] -> weight step (A) = (w_max - w_min)/(k-1) = r*absmax/(k-1),
          (C) = absmax/(k-1), i.e. 2k-1 distinct weights. Equal step: k_A = r*(k-1) + 1, r = (w_max - w_min)/absmax.

Empirical: for every analog tile, seeds 0-4, programming run 0 (noise_seed = 1000*seed, as in the sweep), anchor
setting (t = 1 s) with prog_scale s on a grid: effective weight error e = alpha*w_drift - w_digital (what the forward
pass uses), RMS in units of the tile's absmax. Includes clamping and the unscaled floor.

Writes results/fairness_tiles.csv (analytic, per tile), results/fairness_sigma.csv (empirical, per tile x s),
results/fairness_match.csv ((A) scale with the same pooled RMS error as the (C) points).
Usage: PYTHONPATH=src python scripts/fairness_resolution.py
"""
from __future__ import annotations

import csv
import math
import warnings
from dataclasses import replace

import numpy as np
import pandas as pd
import torch

from analog_geometry.analog import analog_model
from analog_geometry.common import RESULTS, SEEDS
from analog_geometry.sweep import ANCHOR, load_digital

warnings.filterwarnings("ignore")
torch.set_num_threads(12)
S_GRID = [0, 1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]
C_POINTS = {"cifar10": [8.0], "mnist": [32.0, 64.0]}
CONVS = ("single_device", "single_pair")


def tiles(model):
    for name, layer in model.named_analog_layers():
        for i, tile in enumerate(layer.analog_tiles()):
            yield (name if i == 0 else f"{name}.{i}"), tile


def analytic(w: torch.Tensor, nm) -> dict:
    """Per-weight analytic sigmas (prog at scale 1, floor at t = 1 s), RMS in units of absmax."""
    c0, c1 = nm.coeff_dict[nm.acceptance_range]
    gmin, gmax = nm.g_min, nm.g_max
    sp = lambda g: c0 + c1 * g
    K, tr = nm.read_dict["K"], nm.read_dict["t_read"]
    relax = nm.decay_dict["std"][1]  # log(1) = 0
    rd = lambda g: K * torch.log10(g) * math.sqrt(math.log((1 + tr) / (2 * tr)))
    fl = lambda g: torch.sqrt(relax ** 2 + rd(g) ** 2)
    am = w.abs().max()
    wmin, wmax = w.min(), w.max()
    sr_a = (gmax - gmin) / (wmax - wmin)
    ga = gmin + (w - wmin) * sr_a
    sr_c = (gmax - gmin) / am
    gp, gn = gmin + w.clamp(min=0) * sr_c, gmin + (-w).clamp(min=0) * sr_c
    rms = lambda x: float(torch.sqrt((x ** 2).mean()) / am)
    return {"r": float((wmax - wmin) / am),
            "A_prog1": rms(sp(ga) / sr_a), "C_prog1": rms(torch.sqrt(sp(gp) ** 2 + sp(gn) ** 2) / sr_c),
            "A_floor": rms(fl(ga) / sr_a), "C_floor": rms(torch.sqrt(fl(gp) ** 2 + fl(gn) ** 2) / sr_c),
            "A_g_at_w0": float(gmin + (0 - wmin) * sr_a), "C_g_at_w0": gmin}


def main():
    trows, srows = [], []
    for ds in ("mnist", "cifar10"):
        for seed in SEEDS:
            dig, _ = load_digital(ds, seed)
            for s in S_GRID:
                for conv in CONVS:
                    m = analog_model(dig, replace(ANCHOR, prog_scale=float(s)), noise_seed=1000 * seed, converter=conv)
                    for name, tile in tiles(m):
                        nm = tile.rpu_config.noise_model
                        w0 = tile.reference_combined_weights.detach().double()
                        wd = torch.as_tensor(tile.tile.get_weights()).double()
                        alpha = float(tile.alpha.reshape(-1)[0])
                        am = float(w0.abs().max())
                        e = alpha * wd - w0
                        srows.append({"dataset": ds, "seed": seed, "converter": conv, "prog_scale": s, "tile": name,
                                      "n_weights": w0.numel(), "w_absmax": am, "alpha": alpha,
                                      "rms_err_rel": float(torch.sqrt((e ** 2).mean())) / am,
                                      "sse_rel": float(((e / am) ** 2).sum())})
                        if s == 0 and conv == "single_device":
                            trows.append({"dataset": ds, "seed": seed, "tile": name, "n_weights": w0.numel(),
                                          **analytic(w0, nm)})
            print(ds, seed, "done", flush=True)
    pd.DataFrame(trows).to_csv(RESULTS / "fairness_tiles.csv", index=False)
    sg = pd.DataFrame(srows)
    sg.to_csv(RESULTS / "fairness_sigma.csv", index=False)

    # pooled RMS (all weights of the network, each in units of its tile absmax), mean over seeds
    pooled = (sg.groupby(["dataset", "converter", "prog_scale", "seed"])
                .apply(lambda g: math.sqrt(g.sse_rel.sum() / g.n_weights.sum()), include_groups=False)
                .groupby(["dataset", "converter", "prog_scale"]).mean().rename("pooled_rms").reset_index())
    pooled.to_csv(RESULTS / "fairness_pooled.csv", index=False)
    mrows = []
    for ds, pts in C_POINTS.items():
        a = pooled[(pooled.dataset == ds) & (pooled.converter == "single_device")].sort_values("prog_scale")
        c = pooled[(pooled.dataset == ds) & (pooled.converter == "single_pair")].set_index("prog_scale").pooled_rms
        for sc in pts:
            target = float(c.loc[sc])
            sa = float(np.interp(target, a.pooled_rms.values, a.prog_scale.values))  # monotone in s
            mrows.append({"dataset": ds, "s_C": sc, "rms_C": target, "s_A_matched": round(sa, 2),
                          "rms_A_at_sA_grid_interp": target})
    pd.DataFrame(mrows).to_csv(RESULTS / "fairness_match.csv", index=False)
    print(pd.DataFrame(mrows).to_string(index=False))
    print(pooled.pivot_table(index=["dataset", "prog_scale"], columns="converter", values="pooled_rms").to_string())


if __name__ == "__main__":
    main()
