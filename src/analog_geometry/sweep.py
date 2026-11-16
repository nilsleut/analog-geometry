"""Schritt 3: analog sweeps + digital seed-to-seed reference.

Writes
  results/digital_seed_reference_<dataset>.csv  CKA/RSA between all pairs of digital seeds, per layer
  results/analog_sweep_<dataset>.csv            one row per (seed, setting, layer)

Usage: PYTHONPATH=src python -m analog_geometry.sweep --dataset mnist [--seeds 0 1 2 3 4] [--only reference|sweep]
Resumable: settings already present in the CSV are skipped.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import os
import time
import warnings

import numpy as np
import torch

from analog_geometry.analog import AnalogSetting, analog_model
from analog_geometry.common import (
    ACT_DIR, CKPT_DIR, LAYERS, PRE_LAYERS, RESULTS, SEEDS, accuracy, build_model, collect,
    eval_loader, full_loader, get_datasets,
)

PREACT_SWEEPS = ("drift_time", "drift_no_mean")  # ReLU-offset check (decision 2026-11-15)
from analog_geometry.metrics import (
    _centered_gram, linear_cka, linear_cka_stimcentered, rdm, rsa, stimcenter,
)

warnings.filterwarnings("ignore")

# Decisions 2026-11-14 (see notes/step3_design.md): three one-factor sweeps, CMO main model, 2 % acceptance,
# ideal periphery; Wan2022 + aihwkit-default periphery as supplementary points.
ANCHOR = AnalogSetting(device="cmo", prog_scale=1.0, acceptance=2.0, levels=None, t_inference=1.0, periphery="ideal")
PROG_SCALES = [0, 1, 2, 4, 8, 16, 32, 64]          # extend with 128, 256 if CIFAR loss at 64 <= 2 pp
LEVELS = [8, 16, 32, 64, None]                      # without noise and drift (pure quantization)
DAY = 86400.0
T_INF = [1.0, 3600.0, DAY, 7 * DAY, 365 * DAY]      # 1 s, 1 h, 1 d, 1 wk, 1 yr at prog_scale 1
WAN_T = [1.0, DAY, 2 * DAY]
N_REPS = 3                                          # independent programming runs per trained seed


def grid(extra_prog: list[float] = (), prog_scales: list[float] | None = None) -> list[tuple[str, AnalogSetting]]:
    from dataclasses import replace
    off = dict(prog_scale=0.0, drift_scale=0.0, read_scale=0.0, t_inference=0.0)
    g = [("ideal", replace(ANCHOR, **off))]  # sanity: must reproduce digital exactly
    scales = list(PROG_SCALES if prog_scales is None else prog_scales) + list(extra_prog)
    g += [("prog_noise", replace(ANCHOR, prog_scale=float(p))) for p in scales]
    g += [("levels", replace(ANCHOR, levels=k, **off)) for k in LEVELS]
    g += [("drift_time", replace(ANCHOR, t_inference=t)) for t in T_INF]
    g += [("wan2022_time", AnalogSetting(device="wan2022", t_inference=t, periphery="ideal")) for t in WAN_T]
    g += [("periphery_default", replace(ANCHOR, t_inference=DAY, periphery="default"))]
    # offset control (decision 2026-11-15): CMO drift without mean shift
    g += [("drift_no_mean", replace(ANCHOR, t_inference=t, drift_mean=False)) for t in (DAY, 365 * DAY)]
    return g


def setting_key(seed, rep, sweep, row) -> tuple:
    return (str(seed), str(rep), sweep) + tuple(str(row[k]) for k in sorted(row))


def load_digital(name: str, seed: int):
    m = build_model(name)
    m.load_state_dict(torch.load(CKPT_DIR / f"{name}_seed{seed}.pt"))
    acts = dict(np.load(ACT_DIR / f"digital_{name}_seed{seed}.npz"))
    return m, acts


def seed_reference(name: str, seeds: list[int]) -> None:
    rows = []
    acts = {s: load_digital(name, s)[1] for s in seeds}
    for a, b in itertools.combinations(seeds, 2):
        for layer in LAYERS[name]:
            rows.append({"dataset": name, "seed_a": a, "seed_b": b, "layer": layer,
                         "cka": linear_cka(acts[a][layer], acts[b][layer]),
                         "rsa": rsa(acts[a][layer], acts[b][layer]),
                         "cka_stimc": linear_cka_stimcentered(acts[a][layer], acts[b][layer])})
    out = RESULTS / f"digital_seed_reference_{name}.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)


def _pair_metrics(dx, ax, dgram=None, dgram_sc=None, drdm=None) -> dict:
    return {"cka": linear_cka(dx, ax, gram_x=dgram), "rsa": rsa(dx, ax, rdm_x=drdm),
            "cka_stimc": linear_cka_stimcentered(dx, ax, gram_x=dgram_sc)}


def digital_preacts(name: str, seeds: list[int], ev) -> dict:
    out = {}
    for s in seeds:
        m, _ = load_digital(name, s)
        out[s] = collect(m, ev, PRE_LAYERS[name])[1]
    return out


def preact_reference(name: str, dpre: dict) -> None:
    """Seed-to-seed band for pre-activations (all pairs), results/digital_seed_reference_preact_<ds>.csv."""
    rows = []
    for a, b in itertools.combinations(sorted(dpre), 2):
        for layer in PRE_LAYERS[name]:
            rows.append({"dataset": name, "seed_a": a, "seed_b": b, "layer": layer,
                         **_pair_metrics(dpre[a][layer], dpre[b][layer])})
    out = RESULTS / f"digital_seed_reference_preact_{name}.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out, flush=True)


def run_sweep(name: str, seeds: list[int], extra_prog: list[float], prog_scales: list[float] | None = None,
              sweeps: list[str] | None = None, preact: bool = False, converter: str = "single_device",
              times: list[float] | None = None) -> None:
    """converter != single_device -> results/diffpair_<ds>.csv (gate 2026-11-16); times filters t_inference."""
    if converter != "single_device" and preact:
        raise ValueError("--preact only with the default converter")
    _, test = get_datasets(name)
    ev, full = eval_loader(name, test), full_loader(test)
    out = RESULTS / (f"analog_sweep_{name}.csv" if converter == "single_device" else f"diffpair_{name}.csv")
    pre_out = RESULTS / f"preact_{name}.csv"
    setting_cols = list(ANCHOR.as_row())
    head = ["dataset", "seed", "rep", "noise_seed", "sweep"] + setting_cols + ["layer"]
    fields = head + ["cka", "rsa", "cka_stimc", "acc_full", "acc_eval1000", "acc_digital_full", "seconds"]
    pre_fields = head + ["cka", "rsa", "cka_stimc"]

    def _done(path):
        if not path.exists():
            return set()
        with path.open() as fh:
            return {setting_key(r["seed"], r["rep"], r["sweep"], {k: r[k] for k in setting_cols})
                    for r in csv.DictReader(fh)}

    done, pre_done = _done(out), _done(pre_out)
    dpre = {}
    if preact:
        all_seeds = sorted(set(seeds) | set(SEEDS))
        dpre = digital_preacts(name, all_seeds, ev)
        preact_reference(name, {s: dpre[s] for s in SEEDS})
    new, pre_new = not out.exists(), not pre_out.exists()
    t_start, n_new = time.time(), 0
    pre_target = pre_out if preact else os.devnull  # never create an empty, header-less preact file
    with out.open("a", newline="") as fh, open(pre_target, "a", newline="") as pfh:
        w = csv.DictWriter(fh, fieldnames=fields)
        pw = csv.DictWriter(pfh, fieldnames=pre_fields)
        if new:
            w.writeheader()
        if pre_new and preact:
            pw.writeheader()
        for seed in seeds:
            dig, dacts = load_digital(name, seed)
            acc_dig = accuracy(dig, full)
            drdm = {k: rdm(v) for k, v in dacts.items()}
            # fixed digital references, computed once per seed (identical results, less work per run)
            dgram = {k: _centered_gram(v) for k, v in dacts.items()}
            dgram_sc = {k: _centered_gram(stimcenter(v)) for k, v in dacts.items()}
            for sweep, s in grid(extra_prog, prog_scales):
                if sweeps and sweep not in sweeps:
                    continue
                if times and s.t_inference not in times:
                    continue
                row = s.as_row()
                want_pre = preact and sweep in PREACT_SWEEPS
                for rep in range(N_REPS):
                    key = setting_key(seed, rep, sweep, row)
                    need_main = key not in done
                    need_pre = want_pre and key not in pre_done
                    if not (need_main or need_pre):
                        continue
                    noise_seed = 1000 * seed + rep  # distinct per (trained seed, programming run)
                    t0 = time.time()
                    m = analog_model(dig, s, noise_seed=noise_seed, converter=converter)
                    layers = LAYERS[name] + (PRE_LAYERS[name] if need_pre else [])
                    acc_e, acts = collect(m, ev, layers)
                    acc_f = accuracy(m, full)
                    secs = round(time.time() - t0, 1)
                    base = {"dataset": name, "seed": seed, "rep": rep, "noise_seed": noise_seed, "sweep": sweep, **row}
                    if need_pre:
                        for layer in PRE_LAYERS[name]:
                            pw.writerow({**base, "layer": layer, **_pair_metrics(dpre[seed][layer], acts[layer])})
                        pfh.flush()
                    n_new += 1
                    if n_new == 20:
                        rate = (time.time() - t_start) / n_new
                        print(f"RATE after 20 runs: {rate:.1f} s/run wall (incl. metrics)", flush=True)
                    if not need_main:
                        print(f"[{name} seed{seed} rep{rep}] {sweep} preact only acc={acc_f:.4f}", flush=True)
                        continue
                    for layer in LAYERS[name]:
                        w.writerow({"dataset": name, "seed": seed, "rep": rep, "noise_seed": noise_seed,
                                    "sweep": sweep, **row, "layer": layer,
                                    "cka": linear_cka(dacts[layer], acts[layer], gram_x=dgram[layer]),
                                    "rsa": rsa(dacts[layer], acts[layer], rdm_x=drdm[layer]),
                                    "cka_stimc": linear_cka_stimcentered(dacts[layer], acts[layer],
                                                                         gram_x=dgram_sc[layer]),
                                    "acc_full": acc_f, "acc_eval1000": acc_e, "acc_digital_full": acc_dig,
                                    "seconds": secs})
                    fh.flush()
                    print(f"[{name} seed{seed} rep{rep}] {sweep} p={row['prog_scale']} k={row['levels']} "
                          f"t={row['t_inference']} {row['device']}/{row['periphery']} acc={acc_f:.4f} ({secs}s)",
                          flush=True)
    print(f"sweep wall time {time.time() - t_start:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cifar10", "mnist"], required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--only", choices=["reference", "sweep"])
    ap.add_argument("--extra-prog", type=float, nargs="*", default=[],
                    help="additional prog-noise scales, e.g. 128 256 (escalation rule)")
    ap.add_argument("--prog-scales", type=float, nargs="+", default=None,
                    help="override the prog-noise grid (4-h rule: 0 4 16 64)")
    ap.add_argument("--sweeps", nargs="+", default=None, help="run only these sweeps (e.g. ideal prog_noise)")
    ap.add_argument("--preact", action="store_true",
                    help=f"also record pre-activation metrics for {PREACT_SWEEPS} (results/preact_<ds>.csv)")
    ap.add_argument("--converter", choices=["single_device", "single_pair"], default="single_device",
                    help="cmo conductance converter; single_pair writes results/diffpair_<ds>.csv (gate 2026-11-16)")
    ap.add_argument("--times", type=float, nargs="+", default=None, help="only these t_inference values")
    ap.add_argument("--threads", type=int, default=12)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    if args.only != "sweep":
        seed_reference(args.dataset, args.seeds)
    if args.only != "reference":
        run_sweep(args.dataset, args.seeds, args.extra_prog, args.prog_scales, args.sweeps, args.preact,
                  args.converter, args.times)


if __name__ == "__main__":
    main()
