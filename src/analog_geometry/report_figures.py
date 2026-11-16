"""Schritt 4 (after the gate, decision 2026-11-16): report figures + merged table, built ONLY from summary CSVs.

Variants
  (A) single device, drift-free reference   = aihwkit default        -> results/sweep_summary_<ds>.csv (drift_time, ...)
  (B) single device, co-drifting reference  = control 2a (drift_no_mean); see scripts/check_b_equivalence.py
  (C) differential pair G+ - G-             -> results/sweep_summary_diffpair_<ds>.csv (filled in automatically when present)

Writes
  figures/fig1_drift_variants.png/.pdf   drift sweep, columns A | B | C, both datasets
  figures/fig2_noise_levels.png/.pdf     programming noise + levels, A and C (C panels say "ausstehend" until the CSVs exist)
  figures/fig3_delta_w.png/.pdf          measured drift step Δw by weight sign, A vs C (results/gate_drift_offset.csv)
  results/findings_all.csv               findings_<ds>.csv + findings_diffpair_<ds>.csv with a variant column
Usage: PYTHONPATH=src python -m analog_geometry.report_figures
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analog_geometry.analyze import FIG, GRID, INK, INK2, LAYER_STYLE, SURFACE
from analog_geometry.common import LAYERS, RESULTS

DS = [("mnist", "MNIST-MLP"), ("cifar10", "CIFAR-CNN")]
TLAB = {1.0: "1 s", 3600.0: "1 h", 86400.0: "1 d", 604800.0: "1 Wo", 31536000.0: "1 J"}
VARIANT_TITLE = {"A": "(A) Einzelgerät,\ndriftfreie Referenz (Standard)",
                 "B": "(B) Einzelgerät, mitdriftende\nReferenz (= Kontrolle 2a)",
                 "C": "(C) Differenzpaar G⁺−G⁻"}
C_LINE, C_FILL = "#2a78d6", "#eb6834"


def _read(path):
    if not path.exists():
        return None
    s = pd.read_csv(path, keep_default_na=False, na_values=[""], dtype={"levels": str, "drift_mean": str})
    s = s[s["complete"].astype(str) == "True"].copy()
    for c in ("t_inference", "prog_scale"):
        s[c] = s[c].astype(float)
    s["finding"] = s["finding"].astype(str) == "True"
    return s


def load(ds):
    a = _read(RESULTS / f"sweep_summary_{ds}.csv")
    c = _read(RESULTS / f"sweep_summary_diffpair_{ds}.csv")
    band = pd.read_csv(RESULTS / f"reference_band_{ds}.csv").set_index("layer")
    return a, c, band


def drift_rows(a, c, variant):
    """Drift series per variant, incl. the 1 s point (= prog_noise scale 1 at t = 1 s; for B identical to A, ln 1 = 0)."""
    if variant == "A":
        return a[a["sweep"] == "drift_time"]
    if variant == "B":
        one = a[(a["sweep"] == "drift_time") & (a["t_inference"] == 1.0)]
        return pd.concat([one, a[a["sweep"] == "drift_no_mean"]])
    if c is None:
        return None
    one = c[(c["sweep"] == "prog_noise") & (c["prog_scale"] == 1.0) & (c["t_inference"] == 1.0)]
    return pd.concat([one, c[c["sweep"] == "drift_time"]])


def _style(plt):
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.titlecolor": INK, "figure.facecolor": SURFACE,
                         "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def _panel(ax, sub, xcol, ds, band, metric, xform=None):
    """One panel: accuracy loss (network) or a geometry metric per layer with the seed band."""
    if sub is None or sub.empty:
        ax.text(0.5, 0.5, "Lauf ausstehend", transform=ax.transAxes, ha="center", va="center", color=INK2)
        ax.set_xticks([])
        ax.set_yticks([])
        return
    for li, ((color, marker), layer) in enumerate(zip(LAYER_STYLE, LAYERS[ds])):
        g = sub[sub["layer"] == layer].copy()
        g["_x"] = g[xcol].map(xform) if xform else g[xcol]
        g = g.sort_values("_x")
        if metric == "acc_drop_pp":
            if layer != LAYERS[ds][0]:
                continue
            ax.errorbar(g["_x"], g["acc_drop_pp_mean"], yerr=g["acc_drop_pp_std_total"], color=INK2, marker="o", ms=4,
                        lw=1.6, capsize=2)
            f = g[g["finding"]]
            ax.scatter(f["_x"], f["acc_drop_pp_mean"], s=110, facecolors="none", edgecolors=INK, lw=1.4, zorder=5,
                       label="Befund (Vorab-Regel)")
            ax.axhline(1.0, color=INK2, lw=0.9, ls="--")
        else:
            ax.axhspan(band.loc[layer, f"{metric}_min"], band.loc[layer, f"{metric}_max"], color=color, alpha=0.15, lw=0)
            ax.errorbar(g["_x"], g[f"{metric}_mean"], yerr=g[f"{metric}_std_total"], color=color, marker=marker, ms=4,
                        lw=1.6, capsize=2, label=LAYER_LABEL[li])


def _finish(ax):
    ax.grid(True, color=GRID, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)


LAYER_LABEL = ["Schicht 1 (act1)", "Schicht 2 (act2)", "Schicht 3 (fc3 | fc)"]
ROWS = [("acc_drop_pp", "Δ Accuracy [pp]"), ("cka", "linear CKA"), ("rsa", "RSA (Spearman)")]


def fig1():
    import matplotlib.pyplot as plt
    _style(plt)
    fig, axes = plt.subplots(6, 3, figsize=(9.5, 12.5))
    for di, (ds, dname) in enumerate(DS):
        a, c, band = load(ds)
        for vi, v in enumerate("ABC"):
            sub = drift_rows(a, c, v)
            for ri, (m, ylab) in enumerate(ROWS):
                ax = axes[3 * di + ri, vi]
                _panel(ax, sub, "t_inference", ds, band, m)
                if sub is not None and not sub.empty:
                    ax.set_xscale("log")
                    ax.set_xticks(list(TLAB))
                    ax.set_xticklabels(list(TLAB.values()))
                    ax.minorticks_off()
                    ax.set_xlim(0.4, 9e7)
                _finish(ax)
                if di == 0 and ri == 0:
                    ax.set_title(VARIANT_TITLE[v], fontsize=8.5)
                if vi == 0:
                    ax.set_ylabel(f"{dname}\n{ylab}" if ri == 0 else ylab)
    _legend(fig, axes)
    fig.suptitle("Drift-Zeit (CMO, Rauschskalierung 1) – drei Referenz-/Konverter-Annahmen\n"
                 "Mittel ± Std über 5 Seeds × 3 Programmierdurchgänge · jede Achse eigene y-Skala", fontsize=9.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.925))
    _save(fig, "fig1_drift_variants")


def fig2():
    import matplotlib.pyplot as plt
    _style(plt)
    lev_order = ["8", "16", "32", "64", "inf"]
    cols = [("prog_noise", "A", "Programmierrauschen (A)"), ("prog_noise", "C", "Programmierrauschen (C)"),
            ("levels", "A", "Leitwertstufen k (A)"), ("levels", "C", "Leitwertstufen k (C)")]
    fig, axes = plt.subplots(6, 4, figsize=(11.5, 12.5))
    for di, (ds, dname) in enumerate(DS):
        a, c, band = load(ds)
        for ci, (sweep, v, title) in enumerate(cols):
            src = a if v == "A" else c
            sub = None if src is None else src[src["sweep"] == sweep]
            for ri, (m, ylab) in enumerate(ROWS):
                ax = axes[3 * di + ri, ci]
                if sweep == "levels":
                    _panel(ax, sub, "levels", ds, band, m, xform=lambda k: lev_order.index(k))
                    if sub is not None and not sub.empty:
                        ax.set_xticks(range(5))
                        ax.set_xticklabels(["8", "16", "32", "64", "∞"])
                else:
                    _panel(ax, sub, "prog_scale", ds, band, m)
                    if sub is not None and not sub.empty:
                        ax.set_xscale("symlog", linthresh=1)
                        px = sorted(sub["prog_scale"].unique())
                        ax.set_xticks(px)
                        ax.set_xticklabels([f"{p:g}" if p in (0, 1, 4, 16, 64) else "" for p in px])
                        ax.minorticks_off()
                _finish(ax)
                if di == 0 and ri == 0:
                    ax.set_title(title, fontsize=8.5)
                if ci == 0:
                    ax.set_ylabel(f"{dname}\n{ylab}" if ri == 0 else ylab)
    _legend(fig, axes)
    fig.suptitle("Programmierrauschen (Drift 1 s) und Leitwertstufen (ohne Rauschen/Drift): (A) Einzelgerät vs. (C) Differenzpaar\n"
                 "Mittel ± Std über 5 Seeds × 3 Durchgänge · jede Achse eigene y-Skala", fontsize=9.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.925))
    _save(fig, "fig2_noise_levels")


def fig3():
    import matplotlib.pyplot as plt
    _style(plt)
    d = pd.read_csv(RESULTS / "gate_drift_offset.csv")
    d = d[d["sign_class"] != "all"].copy()
    d["rel"] = 100 * d["dw_drift_mean"] / d["w_absmax"]
    d["rel_sd"] = 100 * d["dw_drift_std"] / d["w_absmax"]
    d["rel_exp"] = 100 * d["expected_dw_uniform"] / d["w_absmax"]
    order = ["neg", "zero", "pos"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
    for ax, (ds, dname) in zip(axes, DS):
        g = d[d["dataset"] == ds]
        for k, (conv, lab, col) in enumerate([("single_device", "(A) Einzelgerät", C_LINE),
                                             ("single_pair", "(C) Differenzpaar", C_FILL)]):
            h = g[g["converter"] == conv].groupby("sign_class")[["rel", "rel_sd"]].mean().reindex(order)
            x = np.arange(3) + (k - 0.5) * 0.22
            ax.errorbar(x, h["rel"], yerr=h["rel_sd"], fmt="o" if k == 0 else "s", color=col, ms=5, capsize=3,
                        lw=1.4, label=lab)
        exp = g[g["converter"] == "single_device"]["rel_exp"].mean()
        ax.axhline(exp, color=C_LINE, lw=0.9, ls="--")
        ax.axhline(0, color=INK2, lw=0.6)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["w < 0", "w ≈ 0", "w > 0"])
        ax.set_title(dname, fontsize=8.5)
        _finish(ax)
    axes[0].set_ylabel("Δw (Drift 1 d) [% von max|w|]")
    fig.legend(*axes[0].get_legend_handles_labels(), fontsize=7.5, frameon=False, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 0.86))
    fig.suptitle("Gemessener Drift-Schritt nach Vorzeichen des Gewichts (Mittel; Balken = Std über Gewichte)\n"
                 "gestrichelt: −0,089·ln(t)/scale_ratio (gleichförmiger Versatz, Einzelgerät)", fontsize=8.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.78))
    _save(fig, "fig3_delta_w")


def _legend(fig, axes):
    hs, ls = [], []
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in ls:
                hs.append(h)
                ls.append(l)
    fig.legend(hs, ls, loc="upper center", ncol=len(ls), frameon=False, fontsize=7.5,
               title="Farbband = Spanne der 10 digitalen Seed-Paare je Schicht · gestrichelt: 1 pp (Vorab-Regel)",
               title_fontsize=7.5, bbox_to_anchor=(0.5, 0.955))


def _save(fig, name):
    FIG.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    print("wrote", FIG / f"{name}.png")


def findings_all():
    parts = []
    for ds, _ in DS:
        f = pd.read_csv(RESULTS / f"findings_{ds}.csv", keep_default_na=False, dtype=str)
        f.insert(1, "variant", np.select([f["sweep"] == "drift_no_mean", f["sweep"] == "wan2022_time"],
                                         ["B (2a)", "Wan2022 (Anhang)"], "A"))
        parts.append(f)
        p = RESULTS / f"findings_diffpair_{ds}.csv"
        if p.exists():
            g = pd.read_csv(p, keep_default_na=False, dtype=str)
            g.insert(1, "variant", "C")
            parts.append(g)
    t = pd.concat(parts, ignore_index=True)
    t.to_csv(RESULTS / "findings_all.csv", index=False)
    print("wrote", RESULTS / "findings_all.csv", len(t), "rows")


def main():
    import matplotlib
    matplotlib.use("Agg")
    findings_all()
    fig1()
    fig2()
    fig3()


if __name__ == "__main__":
    main()
