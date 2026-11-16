"""Consistency check: every decimal number in report/report.md against its CSV source (explicit mapping).

SPEC lists, in report order, every number with a decimal comma together with the CSV query that should produce it.
The script walks the report's numbers in order, pairs them with SPEC, rounds the CSV value to the shown number of
decimals (sign ignored) and compares. Reported problems:
  ABWEICHUNG        value from the CSV rounds to a different number
  NICHT ZUGEORDNET  number in the report without a SPEC entry (report and SPEC out of sync -> check stops pairing)
Integers (seeds, pairs, levels, scales, code line numbers) are not checked.

Writes notes/number_check.md. Usage: PYTHONPATH=src python scripts/check_report_numbers.py
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
REPORT = ROOT / "report" / "report.md"
DAY, WEEK, YEAR = 86400.0, 604800.0, 31536000.0
_cache: dict = {}


def csv(name):
    if name not in _cache:
        _cache[name] = pd.read_csv(RES / name, keep_default_na=False, na_values=[""], dtype={"levels": str})
    return _cache[name]


def sel(name, col, **flt):
    d = csv(name)
    for k, v in flt.items():
        d = d[d[k].astype(str) == str(v)] if isinstance(v, str) else d[d[k].astype(float) == float(v)]
    if len(d) != 1:
        raise ValueError(f"{name} {flt}: {len(d)} rows")
    return float(d[col].iloc[0])


def F(var, ds, sweep, col="acc_drop_pp_mean", **flt):
    """findings_all.csv row."""
    return ("findings_all.csv", f"variant={var}, {ds}, {sweep}, {flt} -> {col}",
            lambda: sel("findings_all.csv", col, variant=var, dataset=ds, sweep=sweep, **flt))


def S(ds, sweep, layer, col, **flt):
    """sweep_summary_<ds>.csv row (variant A/B)."""
    f = f"sweep_summary_{ds}.csv"
    return f, f"{sweep}, {layer}, {flt} -> {col}", lambda: sel(f, col, sweep=sweep, layer=layer, **flt)


def B(ds, layer, col, metric="cka"):
    f = f"band_bootstrap_{ds}.csv"
    return f, f"post, {layer}, {metric} -> {col}", lambda: sel(f, col, kind="post", layer=layer, metric=metric)


def R(ds, layer, col):
    f = f"reference_band_{ds}.csv"
    return f, f"{layer} -> {col}", lambda: sel(f, col, layer=layer)


def gate(desc, fn):
    return "gate_drift_offset.csv", desc, fn


def _g():
    g = csv("gate_drift_offset.csv").copy()
    g["rel"] = 100 * g["dw_drift_mean"] / g["w_absmax"]
    return g


def _a_tile_rel():  # (A), class "all", mean over seeds per (dataset, layer)
    g = _g()
    return g[(g.converter == "single_device") & (g.sign_class == "all")].groupby(["dataset", "layer"]).rel.mean()


def _c_sign(cls):
    g = _g()
    return g[(g.converter == "single_pair") & (g.sign_class == cls)].rel.mean()


def _alpha(conv, agg):
    g = _g()
    return getattr(g[g.converter == conv].alpha, agg)()


def _b(ds, t, col):
    b = csv("check_b_equivalence.csv")
    return 100 * b[(b.dataset == ds) & (b.t_inference == t)][col].mean()


def _b_max_meanabs():
    b = csv("check_b_equivalence.csv")
    return 100 * b.groupby(["dataset", "t_inference"]).mean_absdiff_rel.mean().max()


def _mnist_bc(col, agg):
    f = csv("findings_all.csv")
    m = f[(f.dataset == "mnist") & ((f.variant == "B (2a)") |
                                    ((f.variant == "C") & (f.sweep == "drift_time")))]
    return getattr(m[col].astype(float), agg)()


def _mnist_pre_min():  # MLP: every layer before the ReLU, isolated (fc2, fc3) + layer 1 (preact fc1), drift 1 d/1 Wo/1 J
    i = csv("isolated_summary_mnist.csv")
    p = csv("preact_summary_mnist.csv")
    p = p[(p.sweep == "drift_time") & (p.layer == "fc1") & (p.t_inference.astype(float).isin([DAY, WEEK, YEAR]))]
    return min(i[i.stage == "pre"].cka_stimc_mean.min(), p.cka_stimc_mean.min())


CONST = ("aihwkit reram.py:306", "decay_dict['mean'] = -0.08900206", lambda: 0.08900206)
SPEC = [
    # line 7: core statement
    ("2,5", gate("(A) max_Kachel |Δw|/max|w| in % (Mittel über Seeds)", lambda: _a_tile_rel().abs().max())),
    ("30,0", F("A", "cifar10", "drift_time", t_inference=DAY)),
    ("1,1", F("C", "cifar10", "drift_time", t_inference=DAY)),
    ("2,2", F("B (2a)", "cifar10", "drift_no_mean", t_inference=DAY)),
    # line 13: baselines
    ("98,42", ("baseline_summary.csv", "mnist test_acc_mean ×100", lambda: 100 * sel("baseline_summary.csv", "test_acc_mean", dataset="mnist"))),
    ("0,09", ("baseline_summary.csv", "mnist test_acc_std ×100", lambda: 100 * sel("baseline_summary.csv", "test_acc_std", dataset="mnist"))),
    ("76,37", ("baseline_summary.csv", "cifar10 test_acc_mean ×100", lambda: 100 * sel("baseline_summary.csv", "test_acc_mean", dataset="cifar10"))),
    ("0,45", ("baseline_summary.csv", "cifar10 test_acc_std ×100", lambda: 100 * sel("baseline_summary.csv", "test_acc_std", dataset="cifar10"))),
    ("0,089", CONST),
    # line 30: B = 2a
    ("0,2", ("check_b_equivalence.csv", "mnist 1 d frac_differs ×100 (Mittel)", lambda: _b("mnist", DAY, "frac_differs"))),
    ("0,7", ("check_b_equivalence.csv", "cifar10 1 d frac_differs ×100 (Mittel)", lambda: _b("cifar10", DAY, "frac_differs"))),
    ("3,4", ("check_b_equivalence.csv", "mnist 1 J frac_differs ×100 (Mittel)", lambda: _b("mnist", YEAR, "frac_differs"))),
    ("5,8", ("check_b_equivalence.csv", "cifar10 1 J frac_differs ×100 (Mittel)", lambda: _b("cifar10", YEAR, "frac_differs"))),
    ("0,006", ("check_b_equivalence.csv", "max über (ds, t) mean_absdiff_rel ×100 (Obergrenze ≤)", _b_max_meanabs)),
    # line 34: Δw
    ("1,97", gate("(A) min_Kachel |Δw|/max|w| in %", lambda: _a_tile_rel().abs().min())),
    ("2,53", gate("(A) max_Kachel |Δw|/max|w| in %", lambda: _a_tile_rel().abs().max())),
    ("0,089", CONST),
    ("1,33", gate("(C) w>0 Mittel Δw/max|w| in %", lambda: _c_sign("pos"))),
    ("1,33", gate("(C) w<0 Mittel Δw/max|w| in %", lambda: _c_sign("neg"))),
    ("1,04", gate("(C) min alpha", lambda: _alpha("single_pair", "min"))),
    ("1,19", gate("(C) max alpha", lambda: _alpha("single_pair", "max"))),
    ("0,94", gate("(A) min alpha", lambda: _alpha("single_device", "min"))),
    ("1,00", gate("(A) max alpha", lambda: _alpha("single_device", "max"))),
    # line 37-39: CIFAR drift
    ("1,07", F("A", "cifar10", "drift_time", t_inference=1.0)),
    ("19,5", F("A", "cifar10", "drift_time", t_inference=3600.0)),
    ("30,0", F("A", "cifar10", "drift_time", t_inference=DAY)),
    ("35,9", F("A", "cifar10", "drift_time", t_inference=WEEK)),
    ("45,1", F("A", "cifar10", "drift_time", t_inference=YEAR)),
    ("2,2", F("B (2a)", "cifar10", "drift_no_mean", t_inference=DAY)),
    ("3,2", F("B (2a)", "cifar10", "drift_no_mean", t_inference=YEAR)),
    ("1,1", F("C", "cifar10", "drift_time", t_inference=DAY)),
    ("1,4", F("C", "cifar10", "drift_time", t_inference=WEEK)),
    ("2,2", F("C", "cifar10", "drift_time", t_inference=YEAR)),
    ("0,50", F("A", "cifar10", "prog_noise", prog_scale=0.0)),
    ("1,07", F("A", "cifar10", "prog_noise", prog_scale=1.0)),
    # line 44-46: MNIST findings
    ("0,49", F("A", "mnist", "drift_time", t_inference=DAY)),
    ("0,937", S("mnist", "drift_time", "act1", "cka_mean", t_inference=DAY)),
    ("0,956", R("mnist", "act1", "cka_min")),
    ("0,954", S("mnist", "drift_time", "act2", "cka_mean", t_inference=DAY)),
    ("0,962", R("mnist", "act2", "cka_min")),
    ("0,955", S("mnist", "drift_time", "fc3", "cka_mean", t_inference=DAY)),
    ("0,956", R("mnist", "fc3", "cka_min")),
    ("0,85", F("A", "mnist", "drift_time", t_inference=WEEK)),
    ("0,909", S("mnist", "drift_time", "act1", "cka_mean", t_inference=WEEK)),
    ("0,945", S("mnist", "drift_time", "act2", "cka_mean", t_inference=WEEK)),
    ("0,945", S("mnist", "drift_time", "fc3", "cka_mean", t_inference=WEEK)),
    ("0,67", F("A", "mnist", "periphery_default")),
    ("0,935", S("mnist", "periphery_default", "act1", "cka_mean")),
    ("0,950", S("mnist", "periphery_default", "act2", "cka_mean")),
    ("0,948", S("mnist", "periphery_default", "fc3", "cka_mean")),
    # line 48
    ("0,0009", ("sweep_summary_mnist.csv", "drift 1 d fc3 cka_band_min − cka_mean",
                lambda: sel("sweep_summary_mnist.csv", "cka_band_min", sweep="drift_time", layer="fc3", t_inference=DAY)
                - sel("sweep_summary_mnist.csv", "cka_mean", sweep="drift_time", layer="fc3", t_inference=DAY))),
    ("0,0055", S("mnist", "drift_time", "fc3", "cka_edge_tol", t_inference=DAY)),
    ("0,93", F("Wan2022 (Anhang)", "mnist", "wan2022_time", t_inference=DAY)),
    ("0,92", F("Wan2022 (Anhang)", "mnist", "wan2022_time", t_inference=2 * DAY)),
    ("0,08", ("findings_all.csv", "MNIST B + C(Drift): max acc_drop_pp_mean (Obergrenze ≤)", lambda: _mnist_bc("acc_drop_pp_mean", "max"))),
    ("0,997", ("findings_all.csv", "MNIST B + C(Drift): min cka_min_over_layers (Untergrenze ≥)", lambda: _mnist_bc("cka_min_over_layers", "min"))),
    # line 51-56: CKA vs RSA, isolated
    ("0,990", S("mnist", "drift_time", "act1", "rsa_mean", t_inference=DAY)),
    ("0,925", R("mnist", "act1", "rsa_min")),
    ("0,944", R("mnist", "act1", "rsa_max")),
    ("0,9996", ("isolated_summary_mnist.csv + preact_summary_mnist.csv", "min cka_stimc vor ReLU (Untergrenze ≥)", _mnist_pre_min)),
    ("0,898", ("isolated_summary_cifar10.csv", "conv2 pre 1 Wo cka_stimc_mean",
               lambda: sel("isolated_summary_cifar10.csv", "cka_stimc_mean", layer="conv2", stage="pre", t_inference=WEEK))),
    ("0,914", ("isolated_summary_cifar10.csv", "conv2 pre cka_stimc_band_min",
               lambda: sel("isolated_summary_cifar10.csv", "cka_stimc_band_min", layer="conv2", stage="pre", t_inference=WEEK))),
    # line 59-62: noise, levels
    ("11,1", F("A", "cifar10", "prog_noise", prog_scale=4.0)),
    ("0,24", F("A", "mnist", "prog_noise", prog_scale=4.0)),
    ("1,13", F("A", "mnist", "prog_noise", prog_scale=8.0)),
    ("14,7", F("A", "mnist", "prog_noise", prog_scale=16.0)),
    ("18,3", F("A", "cifar10", "levels", levels="8")),
    ("4,4", F("A", "cifar10", "levels", levels="16")),
    ("1,15", F("A", "cifar10", "levels", levels="32")),
    ("0,24", F("A", "cifar10", "levels", levels="64")),
    ("0,46", F("A", "mnist", "levels", levels="8")),
    # variant C, noise and levels (after the night run)
    ("0,25", F("C", "mnist", "prog_noise", prog_scale=16.0)),
    ("1,23", F("C", "mnist", "prog_noise", prog_scale=32.0)),
    ("17,4", F("C", "mnist", "prog_noise", prog_scale=64.0)),
    ("14,7", F("A", "mnist", "prog_noise", prog_scale=16.0)),
    ("69,6", F("A", "mnist", "prog_noise", prog_scale=32.0)),
    ("86,2", F("A", "mnist", "prog_noise", prog_scale=64.0)),
    ("1,41", F("C", "cifar10", "prog_noise", prog_scale=4.0)),
    ("4,89", F("C", "cifar10", "prog_noise", prog_scale=8.0)),
    ("16,2", F("C", "cifar10", "prog_noise", prog_scale=16.0)),
    ("11,1", F("A", "cifar10", "prog_noise", prog_scale=4.0)),
    ("32,0", F("A", "cifar10", "prog_noise", prog_scale=8.0)),
    ("54,7", F("A", "cifar10", "prog_noise", prog_scale=16.0)),
    ("8,44", F("C", "cifar10", "levels", levels="8")),
    ("1,50", F("C", "cifar10", "levels", levels="16")),
    ("0,44", F("C", "cifar10", "levels", levels="32")),
    ("0,07", F("C", "cifar10", "levels", levels="64")),
    ("18,3", F("A", "cifar10", "levels", levels="8")),
    ("4,4", F("A", "cifar10", "levels", levels="16")),
    ("1,15", F("A", "cifar10", "levels", levels="32")),
    ("0,24", F("A", "cifar10", "levels", levels="64")),
    ("0,26", F("C", "mnist", "levels", levels="8")),
    ("0,46", F("A", "mnist", "levels", levels="8")),
    # line 73: limits, bootstrap
    ("0,9557", B("mnist", "act1", "band_min_observed")),
    ("0,9557", B("mnist", "act1", "boot_ci_low_2.5")),
    ("0,9631", B("mnist", "act1", "boot_ci_high_97.5")),
]
# numbers added to the report later are appended via EXTRA (see bottom of file)


def main(spec=None):
    spec = spec or SPEC
    text = REPORT.read_text(encoding="utf-8")
    found = [(ln, m.group(0), line[max(0, m.start() - 35):m.end() + 12].replace("|", "/").strip())
             for ln, line in enumerate(text.splitlines(), 1) for m in re.finditer(r"\d+,\d+", line)]
    rows, n_bad = [], 0
    for i, (ln, tok, ctx) in enumerate(found):
        if i >= len(spec) or spec[i][0] != tok:
            exp = spec[i][0] if i < len(spec) else "-"
            rows.append((ln, tok, f"NICHT ZUGEORDNET (Spec erwartet {exp}); Prüfung ab hier abgebrochen", "-", "-", ctx))
            n_bad += 1
            break
        f, desc, fn = spec[i][1]
        try:
            v = fn()
        except Exception as e:  # noqa: BLE001
            rows.append((ln, tok, f"FEHLER {e}", f, desc, ctx))
            n_bad += 1
            continue
        d = len(tok.split(",")[1])
        shown = float(tok.replace(",", "."))
        ok = round(abs(v), d) == round(shown, d)
        if not ok:
            n_bad += 1
        rows.append((ln, tok, "ok" if ok else f"ABWEICHUNG: CSV {abs(v):.{d + 2}f}", f, f"{desc} = {v:.6g}", ctx))
    extra = len(spec) - len(found)
    out = ROOT / "notes" / "number_check.md"
    lines = ["# Zahlenprüfung report/report.md -> results/*.csv",
             "",
             "Erzeugt von `scripts/check_report_numbers.py` (feste Zuordnung Zahl -> CSV-Abfrage, in Reihenfolge des Reports).",
             "Geprüft: jede Dezimalzahl, Vorzeichen ignoriert, gerundet auf die angezeigten Nachkommastellen.",
             "Nicht geprüft: Ganzzahlen (Seeds, Paare, Stufen, Skalen, Zeilennummern im Code).",
             "Bei „Obergrenze ≤“ bzw. „Untergrenze ≥“ steht im Report eine Schranke; geprüft wird der gerundete Extremwert.",
             "",
             f"**Ergebnis: {len(found)} Zahlen im Report, {n_bad} Probleme"
             + (f", {extra} Spec-Einträge ohne Gegenstück im Report" if extra > 0 else "") + ".**",
             "",
             "| Zeile | Zahl | Status | Datei | Abfrage = Wert | Kontext |", "|---|---|---|---|---|---|"]
    lines += [f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} |" for r in rows]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(found)} numbers, {n_bad} problems, {max(extra, 0)} unused spec entries -> {out}")
    for r in rows:
        if r[2] != "ok":
            print("  line", r[0], r[1], r[2], "|", r[4])


EXTRA: list = []

if __name__ == "__main__":
    main(SPEC + EXTRA)
