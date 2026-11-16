"""Is control 2a (drift without mean term) the same as variant (B) "single device, reference drifts along"?

Ideal co-drifting reference: every device and the zero reference shift by the same mean dg = -0.089 ln t, so the read-out
weight is w_A - dg/scale_ratio (A = standard model, same random draws). 2a instead sets dg := 0 before the clamp to [g_min, g_max].
Both use identical random numbers for the same noise_seed (the mean term does not consume RNG draws), so any difference
between (w_A - dg/sr) and w_2a comes ONLY from the clamp. Not modelled in 2a: noise of the reference device itself.

Writes results/check_b_equivalence.csv (per dataset, seed, tile, t).
Usage: PYTHONPATH=src python scripts/check_b_equivalence.py
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
torch.set_num_threads(4)


def tiles(model):
    for name, layer in model.named_analog_layers():
        for i, tile in enumerate(layer.analog_tiles()):
            yield (name if i == 0 else f"{name}.{i}"), tile


def main():
    rows = []
    for ds in ("mnist", "cifar10"):
        for seed in SEEDS:
            dig, _ = load_digital(ds, seed)
            for t in (DAY, 365 * DAY):
                a = dict(tiles(analog_model(dig, replace(ANCHOR, t_inference=t), noise_seed=1000 * seed)))
                b = dict(tiles(analog_model(dig, replace(ANCHOR, t_inference=t, drift_mean=False), noise_seed=1000 * seed)))
                for name, ta in a.items():
                    tb = b[name]
                    nm = ta.rpu_config.noise_model
                    wp_a, wp_b = ta.programmed_weights.double(), tb.programmed_weights.double()
                    assert torch.equal(wp_a, wp_b), "programming draws differ"
                    sr = (nm.g_max - nm.g_min) / float(wp_a.max() - wp_a.min())
                    dg = nm.decay_dict["mean"][0] * math.log(t) * nm.drift_scale
                    wa = torch.as_tensor(ta.tile.get_weights()).double() - dg / sr  # A read against a co-drifting reference
                    wb = torch.as_tensor(tb.tile.get_weights()).double()
                    absmax = float(wp_a.abs().max())
                    diff = (wa - wb).abs()
                    rows.append({"dataset": ds, "seed": seed, "layer": name, "t_inference": t, "n_weights": diff.numel(),
                                 "g_shift_uS": dg, "frac_differs": float((diff > 1e-4 * absmax).double().mean()),  # float32 tile storage -> tolerance
                                 "mean_absdiff_rel": float(diff.mean()) / absmax,
                                 "max_absdiff_rel": float(diff.max()) / absmax})
            print(ds, seed, "done", flush=True)
    out = RESULTS / "check_b_equivalence.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
