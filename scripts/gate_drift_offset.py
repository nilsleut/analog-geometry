"""Gate before step 4 (2026-11-16): does the CMO drift act on the weight (uniform offset) or on a device difference?

For every analog layer, seeds 0-4, programming run 0 (noise_seed = 1000*seed, as in the sweep), drift 1 d, anchor setting:
  dw_drift = w_drifted - w_programmed   (pure drift step, what the gate asks about)
  dw_total = w_drifted - w_digital      (programming + drift)
split by the sign of the digital weight: pos (w > eps), neg (w < -eps), zero (|w| <= eps), eps = 1 % of max|w|.
expected_dw = mean conductance shift / scale_ratio, computed from the converter's own mapping
(single_device: scale_ratio = (g_max - g_min) / (w_max - w_min) of the programmed weights).
Both converters: "single_device" (aihwkit default for CMO) and "single_pair" (G+ - G-, own override).

Writes results/gate_drift_offset.csv.
Usage: PYTHONPATH=src python scripts/gate_drift_offset.py
"""
from __future__ import annotations

import csv
import math
import warnings
from dataclasses import replace

import torch

from analog_geometry.analog import analog_model
from analog_geometry.common import RESULTS, SEEDS
from analog_geometry.sweep import ANCHOR, DAY, load_digital

warnings.filterwarnings("ignore")
torch.set_num_threads(8)
T = DAY


def tiles(model):
    for name, layer in model.named_analog_layers():
        for i, tile in enumerate(layer.analog_tiles()):
            yield (name if i == 0 else f"{name}.{i}"), tile


def main():
    s = replace(ANCHOR, t_inference=T)
    rows = []
    for ds in ("mnist", "cifar10"):
        for seed in SEEDS:
            dig, _ = load_digital(ds, seed)
            for conv in ("single_device", "single_pair"):
                m = analog_model(dig, s, noise_seed=1000 * seed, converter=conv)
                for name, tile in tiles(m):
                    nm = tile.rpu_config.noise_model
                    w0 = tile.reference_combined_weights.detach().double()
                    wp = tile.programmed_weights.detach().double()
                    wd = torch.as_tensor(tile.tile.get_weights()).double()
                    alpha = float(tile.alpha.reshape(-1)[0])
                    dg = nm.decay_dict["mean"][0] * math.log(T) * nm.drift_scale  # mean conductance shift (uS)
                    if conv == "single_device":
                        sr = (nm.g_max - nm.g_min) / float(wp.max() - wp.min())
                    else:
                        sr = (nm.g_max - nm.g_min) / float(wp.abs().max())
                    eps = 0.01 * float(w0.abs().max())
                    for cls, mask in (("pos", w0 > eps), ("neg", w0 < -eps), ("zero", w0.abs() <= eps), ("all", w0 == w0)):
                        dd, dt = (wd - wp)[mask], (wd - w0)[mask]
                        de = (alpha * wd - w0)[mask]
                        rows.append({"dataset": ds, "seed": seed, "converter": conv, "layer": name, "t_inference": T,
                                     "sign_class": cls, "n_weights": int(mask.sum()), "w_absmax": float(w0.abs().max()),
                                     "dw_drift_mean": float(dd.mean()), "dw_drift_std": float(dd.std()),
                                     "dw_total_mean": float(dt.mean()), "dw_total_std": float(dt.std()),
                                     "dw_eff_alpha_mean": float(de.mean()), "dw_eff_alpha_std": float(de.std()),
                                     "alpha": alpha, "g_shift_uS": dg, "scale_ratio_uS_per_w": sr,
                                     "expected_dw_uniform": dg / sr})
            print(ds, seed, "done", flush=True)
    out = RESULTS / "gate_drift_offset.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
