"""Schritt 4: aggregate CSVs -> summary CSVs + main figure. Every number in report/figures comes from here.

Reads   results/baseline_accuracy.csv, results/digital_seed_reference_<ds>.csv, results/analog_sweep_<ds>.csv
Writes  results/baseline_summary.csv
        results/reference_band_<ds>.csv
        results/sweep_summary_<ds>.csv
        figures/main_<ds>.png / .pdf

Variance decomposition per (setting, layer), 5 trained seeds x 3 programming runs:
  std_total        std over all 15 values (ddof=1)
  std_between_seed std over the 5 per-seed means (ddof=1)
  std_within_seed  sqrt(mean over seeds of the variance over the 3 runs (ddof=1))  = programming-run variability
Reference band = [min, max] over the 10 digital seed pairs. "outside_band" = analog mean below the band minimum.

Usage: PYTHONPATH=src python -m analog_geometry.analyze --dataset mnist
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from analog_geometry.common import LAYERS, RESULTS, ROOT

FIG = ROOT / "figures"
# dataviz reference palette, categorical slots 1-3 (validated all-pairs, light mode) + marker as 2nd encoding
LAYER_STYLE = [("#2a78d6", "o"), ("#eb6834", "s"), ("#1baf7a", "^")]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

SWEEP_X = {"prog_noise": "prog_scale", "levels": "levels", "drift_time": "t_inference", "wan2022_time": "t_inference"}
SETTING_COLS = ["sweep", "device", "prog_scale", "drift_scale", "read_scale", "acceptance", "drift_mean", "levels",
                "t_inference", "periphery"]
GEOM = ["cka", "rsa", "cka_stimc"]  # cka_stimc = offset control (interpretation only, not part of the finding rule)


def baseline_summary() -> pd.DataFrame:
    b = pd.read_csv(RESULTS / "baseline_accuracy.csv")
    s = b.groupby("dataset").agg(n_seeds=("seed", "count"),
                                 test_acc_mean=("test_acc_full", "mean"), test_acc_std=("test_acc_full", "std"),
                                 eval1000_acc_mean=("test_acc_eval1000", "mean"),
                                 eval1000_acc_std=("test_acc_eval1000", "std")).reset_index()
    s.to_csv(RESULTS / "baseline_summary.csv", index=False)
    return s


def reference_band(ds: str) -> pd.DataFrame:
    r = pd.read_csv(RESULTS / f"digital_seed_reference_{ds}.csv")
    agg = {"n_pairs": ("cka", "count")}
    for m in GEOM:
        agg.update({f"{m}_mean": (m, "mean"), f"{m}_std": (m, "std"), f"{m}_min": (m, "min"), f"{m}_max": (m, "max")})
    band = r.groupby("layer").agg(**agg).reindex(LAYERS[ds]).reset_index()
    band.insert(0, "dataset", ds)
    band.to_csv(RESULTS / f"reference_band_{ds}.csv", index=False)
    return band


def _decompose(g: pd.DataFrame, col: str) -> dict:
    per_seed = g.groupby("seed")[col]
    within = per_seed.var(ddof=1).mean()
    return {f"{col}_mean": g[col].mean(), f"{col}_std_total": g[col].std(ddof=1),
            f"{col}_std_between_seed": per_seed.mean().std(ddof=1),
            f"{col}_std_within_seed": float(np.sqrt(within)) if pd.notna(within) else np.nan}


def sweep_summary(ds: str, band: pd.DataFrame) -> pd.DataFrame:
    # read levels as text: otherwise pandas parses "inf" as float and "8" becomes "8.0"
    d = pd.read_csv(RESULTS / f"analog_sweep_{ds}.csv", keep_default_na=False, na_values=[""],
                    dtype={"levels": str, "drift_mean": str})
    d["acc_drop_pp"] = 100 * (d["acc_digital_full"] - d["acc_full"])
    rows = []
    for key, g in d.groupby(SETTING_COLS + ["layer"], dropna=False, sort=False):
        row = dict(zip(SETTING_COLS + ["layer"], key))
        row.update(n=len(g), n_seeds=g["seed"].nunique(), n_reps=g["rep"].nunique())
        for col in GEOM + ["acc_full", "acc_drop_pp"]:
            row.update(_decompose(g, col))
        b = band.set_index("layer").loc[row["layer"]]
        for m in GEOM:
            row[f"{m}_band_min"] = b[f"{m}_min"]
            row[f"{m}_outside_band"] = bool(row[f"{m}_mean"] < b[f"{m}_min"])
        rows.append(row)
    s = pd.DataFrame(rows)
    s.insert(0, "dataset", ds)
    # only settings with all 5 seeds x 3 runs count (figure + finding rule); partial ones stay visible in the CSV
    s["complete"] = (s["n_seeds"] == 5) & (s["n_reps"] == 3) & (s["n"] == 15)
    # Pre-registered finding rule (notes/step3_design.md, 2026-11-15), evaluated per setting over layers
    keys = [c for c in SETTING_COLS]
    per = s[s["complete"]].groupby(keys, dropna=False, sort=False).agg(
        acc_drop=("acc_drop_pp_mean", "first"),
        any_cka=("cka_outside_band", "any"), any_rsa=("rsa_outside_band", "any")).reset_index()
    per["acc_stable"] = per["acc_drop"] <= 1.0
    per["finding"] = per["acc_stable"] & (per["any_cka"] | per["any_rsa"])
    per["finding_metric"] = np.where(~per["finding"], "",
                                     np.where(per["any_cka"] & per["any_rsa"], "CKA+RSA",
                                              np.where(per["any_cka"], "nur CKA", "nur RSA")))
    s = s.merge(per[keys + ["acc_stable", "finding", "finding_metric"]], on=keys, how="left")
    s["finding"] = s["finding"].fillna(False).astype(bool)
    s.to_csv(RESULTS / f"sweep_summary_{ds}.csv", index=False)
    return s


def findings_table(ds: str, s: pd.DataFrame) -> pd.DataFrame:
    """One row per complete setting: accuracy loss, layers outside the band, pre-registered rule. -> results/findings_<ds>.csv"""
    rows = []
    for key, g in s[s["complete"]].groupby(SETTING_COLS, dropna=False, sort=False):
        r0 = g.iloc[0]
        row = dict(zip(SETTING_COLS, key))
        row.update(acc_drop_pp_mean=r0["acc_drop_pp_mean"],
                   acc_drop_pp_std_between_seed=r0["acc_drop_pp_std_between_seed"],
                   acc_drop_pp_std_within_seed=r0["acc_drop_pp_std_within_seed"],
                   acc_stable=bool(r0["acc_drop_pp_mean"] <= 1.0))
        for m in GEOM:
            row[f"{m}_layers_outside"] = ";".join(g.loc[g[f"{m}_outside_band"], "layer"]) or "-"
            row[f"{m}_min_over_layers"] = g[f"{m}_mean"].min()
        row["finding"] = bool(r0["finding"])
        row["finding_metric"] = r0["finding_metric"] if row["finding"] else ""
        rows.append(row)
    t = pd.DataFrame(rows)
    t.insert(0, "dataset", ds)
    t.to_csv(RESULTS / f"findings_{ds}.csv", index=False)
    return t


PRE_TO_POST = {"cifar10": {"conv1": "act1", "conv2": "act2"}, "mnist": {"fc1": "act1", "fc2": "act2"}}


def preact_summary(ds: str, post: pd.DataFrame) -> pd.DataFrame | None:
    """ReLU-offset check: metrics on pre-activations vs. own seed band, side by side with the post-ReLU layer."""
    p_path, r_path = RESULTS / f"preact_{ds}.csv", RESULTS / f"digital_seed_reference_preact_{ds}.csv"
    if not (p_path.exists() and r_path.exists()):
        return None
    d = pd.read_csv(p_path, keep_default_na=False, na_values=[""], dtype={"levels": str, "drift_mean": str})
    ref = pd.read_csv(r_path)
    band = ref.groupby("layer").agg(**{f"{m}_band_min": (m, "min") for m in GEOM}).reset_index()
    rows = []
    for key, g in d.groupby(SETTING_COLS + ["layer"], dropna=False, sort=False):
        row = dict(zip(SETTING_COLS + ["layer"], key))
        row.update(n=len(g), n_seeds=g["seed"].nunique(), n_reps=g["rep"].nunique())
        for col in GEOM:
            row.update(_decompose(g, col))
        b = band.set_index("layer").loc[row["layer"]]
        for m in GEOM:
            row[f"{m}_band_min"] = b[f"{m}_band_min"]
            row[f"{m}_outside_band"] = bool(row[f"{m}_mean"] < b[f"{m}_band_min"])
        # matching post-ReLU layer from the main summary
        q = post[(post["sweep"] == row["sweep"]) & (post["layer"] == PRE_TO_POST[ds][row["layer"]]) &
                 (post["t_inference"].astype(float) == float(row["t_inference"]))]
        row["post_layer"] = PRE_TO_POST[ds][row["layer"]]
        for m in GEOM:
            row[f"post_{m}_mean"] = q[f"{m}_mean"].iloc[0] if len(q) == 1 else np.nan
            row[f"post_{m}_outside_band"] = bool(q[f"{m}_outside_band"].iloc[0]) if len(q) == 1 else None
        rows.append(row)
    s = pd.DataFrame(rows)
    s.insert(0, "dataset", ds)
    s.to_csv(RESULTS / f"preact_summary_{ds}.csv", index=False)
    return s


def _xvals(sub: pd.DataFrame, sweep: str):
    col = SWEEP_X[sweep]
    if sweep == "levels":
        order = ["8", "16", "32", "64", "inf"]
        sub = sub.set_index("levels").reindex(order).reset_index()
        return sub, np.arange(len(order)), ["8", "16", "32", "64", "∞"]
    sub = sub.sort_values(col)
    return sub, sub[col].to_numpy(dtype=float), None


def main_figure(ds: str, s: pd.DataFrame, band: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.titlecolor": INK, "figure.facecolor": SURFACE,
                         "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})
    sweeps = [("prog_noise", "Programmierrauschen (Skalierung)\nDrift 1 s"),
              ("levels", "Leitwertstufen k\nohne Rauschen/Drift"),
              ("drift_time", "Drift-Zeit\nRauschskalierung 1")]
    metrics = [("cka", "Linear CKA vs. digital"), ("rsa", "RSA (Spearman) vs. digital"),
               ("acc_drop_pp", "Accuracy-Verlust [pp]")]
    s = s[s["complete"]]
    fig, axes = plt.subplots(3, 3, figsize=(10, 8.2))
    for r in (0, 1):  # geometry rows share y; accuracy row has its own y-axis per sweep
        for c in (1, 2):
            axes[r, c].sharey(axes[r, 0])
            axes[r, c].tick_params(labelleft=False)
    tlabels = {1.0: "1 s", 3600.0: "1 h", 86400.0: "1 d", 604800.0: "1 Wo", 31536000.0: "1 J"}
    for c, (sweep, title) in enumerate(sweeps):
        for r, (m, ylabel) in enumerate(metrics):
            ax = axes[r, c]
            for (color, marker), layer in zip(LAYER_STYLE, LAYERS[ds]):
                sub = s[(s["sweep"] == sweep) & (s["layer"] == layer)]
                sub, x, ticks = _xvals(sub, sweep)
                if m != "acc_drop_pp":
                    b = band.set_index("layer").loc[layer]
                    ax.axhspan(b[f"{m}_min"], b[f"{m}_max"], color=color, alpha=0.13, lw=0)
                elif layer != LAYERS[ds][0]:
                    continue  # accuracy is per network, plot once
                ax.errorbar(x, sub[f"{m}_mean"], yerr=sub[f"{m}_std_total"], color=color if m != "acc_drop_pp" else INK2,
                            marker=marker, ms=5, lw=2, capsize=2, label=layer if m != "acc_drop_pp" else "Netz")
            if sweep == "prog_noise":
                ax.set_xscale("symlog", linthresh=1)
                px = sorted(s.loc[s["sweep"] == "prog_noise", "prog_scale"].astype(float).unique())
                ax.set_xticks(px)
                ax.set_xticklabels([f"{p:g}" for p in px])
            elif sweep == "levels":
                ax.set_xticks(x)
                ax.set_xticklabels(ticks)
            else:
                ax.set_xscale("log")
                ax.set_xticks(list(tlabels))
                ax.set_xticklabels(list(tlabels.values()))
                ax.minorticks_off()
            if m == "acc_drop_pp":
                ax.axhline(1.0, color=INK2, lw=1, ls="--")
                ax.annotate("1 pp (Vorab-Regel)", xy=(0.02, 1.0), xycoords=("axes fraction", "data"),
                            xytext=(0, 3), textcoords="offset points", fontsize=7, color=INK2)
            ax.grid(True, color=GRID, lw=0.6)
            ax.spines[["top", "right"]].set_visible(False)
            if r == 0:
                ax.set_title(title, fontsize=9)
            if c == 0:
                ax.set_ylabel(ylabel)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Schicht · Band gleicher Farbe = Spanne der 10 digitalen Seed-Paare (min–max)",
               fontsize=8, title_fontsize=8, frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle(f"{ds}: Geometrie-Abweichung vs. Nichtidealität (ReRAM CMO, Mittel ± Std über 5 Seeds × 3 Durchgänge)",
                 fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    FIG.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"main_{ds}.{ext}", dpi=200)
    print("wrote", FIG / f"main_{ds}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cifar10", "mnist"], required=True)
    args = ap.parse_args()
    baseline_summary()
    band = reference_band(args.dataset)
    s = sweep_summary(args.dataset, band)
    findings_table(args.dataset, s)
    preact_summary(args.dataset, s)
    main_figure(args.dataset, s, band)


if __name__ == "__main__":
    main()
