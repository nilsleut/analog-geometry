"""Layer-isolated ReLU test (decision 2026-11-15/16).

For every layer l >= 2: feed the DIGITAL activations of layer l-1 into the ANALOG (programmed + drifted) layer l and
compare to the digital layer l on the same input -- before the ReLU (pre) and after it (post, if the layer has one).
The analog network is created exactly as in the sweep (same setting, same noise_seed), so its weights are the same draw.

Writes results/isolated_<ds>.csv (one row per seed, rep, setting, layer, stage).
Usage: PYTHONPATH=src python -m analog_geometry.isolated --dataset mnist
"""
from __future__ import annotations

import argparse
import csv
import warnings
from dataclasses import replace

import torch

from analog_geometry.analog import analog_model
from analog_geometry.common import RESULTS, SEEDS, eval_loader, get_datasets
from analog_geometry.metrics import linear_cka, linear_cka_stimcentered, rsa
from analog_geometry.sweep import ANCHOR, DAY, N_REPS, load_digital

warnings.filterwarnings("ignore")
T_LIST = [DAY, 7 * DAY, 365 * DAY]


@torch.no_grad()
def layer_io(name: str, model, x: torch.Tensor) -> dict:
    """Digital forward pass that records the input of each layer l>=2 and its pre/post outputs.

    Keys: '<layer>_in', '<layer>_pre', '<layer>_post' (post only where a ReLU follows).
    """
    model.eval()
    d = {}
    if name == "mnist":
        h1 = model.act1(model.fc1(torch.flatten(x, 1)))
        d["fc2_in"] = h1
        d["fc2_pre"] = model.fc2(h1)
        d["fc2_post"] = model.act2(d["fc2_pre"])
        d["fc3_in"] = d["fc2_post"]
        d["fc3_pre"] = model.fc3(d["fc3_in"])
    else:
        h1 = model.pool1(model.act1(model.conv1(x)))
        d["conv2_in"] = h1
        d["conv2_pre"] = model.conv2(h1)
        d["conv2_post"] = model.act2(d["conv2_pre"])
        d["fc_in"] = torch.flatten(model.pool2(d["conv2_post"]), 1)
        d["fc_pre"] = model.fc(d["fc_in"])
    return d


@torch.no_grad()
def isolated_outputs(name: str, analog, dig_io: dict) -> dict:
    """Analog layer l on the DIGITAL input of layer l."""
    analog.eval()
    out = {}
    layers = [("fc2", "act2"), ("fc3", None)] if name == "mnist" else [("conv2", "act2"), ("fc", None)]
    for layer, relu in layers:
        pre = getattr(analog, layer)(dig_io[f"{layer}_in"])
        out[f"{layer}_pre"] = pre
        if relu:
            out[f"{layer}_post"] = getattr(analog, relu)(pre)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cifar10", "mnist"], required=True)
    ap.add_argument("--threads", type=int, default=12)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    name = args.dataset
    _, test = get_datasets(name)
    x = torch.cat([b[0] for b in eval_loader(name, test)])
    settings = [("drift_time", replace(ANCHOR, t_inference=t)) for t in T_LIST]
    rows = []
    for seed in SEEDS:
        dig, _ = load_digital(name, seed)
        dio = layer_io(name, dig, x)
        for sweep, s in settings:
            for rep in range(N_REPS):
                noise_seed = 1000 * seed + rep  # same draw as in sweep.py
                ana = analog_model(dig, s, noise_seed=noise_seed)
                aio = isolated_outputs(name, ana, dio)
                for key, A in aio.items():
                    layer, stage = key.rsplit("_", 1)
                    D = dio[key].flatten(1).numpy()
                    A = A.flatten(1).numpy()
                    rows.append({"dataset": name, "seed": seed, "rep": rep, "noise_seed": noise_seed, "sweep": sweep,
                                 **s.as_row(), "layer": layer, "stage": stage,
                                 "cka": linear_cka(D, A), "cka_stimc": linear_cka_stimcentered(D, A), "rsa": rsa(D, A)})
            print(f"[{name} seed{seed}] {sweep} t={s.t_inference:g} done", flush=True)
    out = RESULTS / f"isolated_{name}.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
