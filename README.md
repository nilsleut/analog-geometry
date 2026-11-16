# analog-geometry

Dieses Projekt untersucht, wie Nichtidealitäten der analogen In-Memory-Inferenz die Repräsentationsgeometrie kleiner, digital trainierter Netze verändern. Das sind Programmierrauschen, endliche Leitwertstufen und Drift (IBM aihwkit, `ReRamCMONoiseModel`). Gemessen werden Accuracy sowie linear CKA und RSA je Schicht, jeweils gegen ein Referenzband aus 10 Paaren digitaler Seeds. Die Einordnung folgt einer vorab festgelegten Befund-Regel.

Kernergebnis: Unter den untersuchten Einstellungen hängt der Drift-Schaden überwiegend an der Konverter-Annahme des Modells (Einzelgerät mit driftfreier Referenz gegenüber Differenzpaar). Details stehen in [report/report.md](report/report.md).

## Umgebung

- WSL2 Ubuntu 24.04, Python 3.12.3, CPU only.
- torch 2.15.0+cpu, torchvision 0.29.2+cpu, aihwkit 1.2.0, numpy 2.5.4, scipy 1.18.1, pandas 3.1.1, matplotlib 3.11.2. Die genauen Pins stehen in [requirements.txt](requirements.txt).
- Natives Windows funktioniert nicht, weil der aihwkit-Quell-Build an OpenBLAS scheitert. Dazu, zu den Umgehungen von aihwkit-Bugs (im eigenen Code, das Paket bleibt ungepatcht) und zum Bezug auf IBM/aihwkit#828 siehe [ENVIRONMENT.md](ENVIRONMENT.md).

```bash
# in WSL, im Repo-Wurzelverzeichnis
python3 -m venv ~/.venvs/analog-geometry
~/.venvs/analog-geometry/bin/pip install -r requirements.txt
export AG_DATA_DIR=~/data/analog-geometry PYTHONPATH=src     # Datensätze werden dorthin geladen
alias py=~/.venvs/analog-geometry/bin/python
py scripts/smoke_test_aihwkit.py                               # -> results/env_smoke_test.csv
py -m pytest -q                                                # Metrik- und Augmentierungstests
```

## Reproduktion

Alle Läufe verwenden 5 Trainings-Seeds × 3 Programmierdurchgänge (`noise_seed = 1000·seed + Durchgang`). Die Sweeps sind fortsetzbar: Bereits vorhandene Einstellungen in der CSV werden übersprungen. Laufzeiten auf 14 CPU-Threads: MNIST etwa 2–4 s pro Lauf, CIFAR etwa 30–50 s.

| Schritt / Abbildung | Befehl (in WSL) | Ergebnis |
|---|---|---|
| Baselines | `py -m analog_geometry.train_baseline --dataset mnist` (ebenso `cifar10`) | `results/baseline_accuracy.csv`, Checkpoints, Aktivierungen |
| Seed-Referenz + Sweep (A) | `py -m analog_geometry.sweep --dataset mnist` | `results/digital_seed_reference_<ds>.csv`, `results/analog_sweep_<ds>.csv` |
| Prä-Aktivierungen (Drift) | `py -m analog_geometry.sweep --dataset mnist --only sweep --sweeps drift_time drift_no_mean --preact` | `results/preact_<ds>.csv` |
| CIFAR-Restlauf (Reihenfolge wie durchgeführt) | `bash scripts/run_cifar_rest.sh` (Git Bash auf dem Host) | committet je Sweep |
| Isolierter Schichttest | `py -m analog_geometry.isolated --dataset mnist` | `results/isolated_<ds>.csv` |
| Gate: Δw nach Vorzeichen (Abb. 3) | `py scripts/gate_drift_offset.py` | `results/gate_drift_offset.csv` |
| Gate: (B) = 2a? | `py scripts/check_b_equivalence.py` | `results/check_b_equivalence.csv` |
| Variante (C) Drift | `bash scripts/run_diffpair.sh` (in WSL) | `results/diffpair_<ds>.csv` |
| Variante (C) Rauschen + Stufen | `bash scripts/run_diffpair_grid.sh` (Git Bash auf dem Host) | `results/diffpair_<ds>.csv` |
| Fairness (C) vs. (A): effektive Auflösung | `py scripts/fairness_resolution.py`, `py scripts/fairness_levels_error.py` | `results/fairness_tiles.csv`, `fairness_sigma.csv`, `fairness_pooled.csv`, `fairness_match.csv`, `fairness_levels_error.csv` |
| Fairness: (A) an angeglichenen Punkten | `py -m analog_geometry.sweep --dataset mnist --only sweep --sweeps prog_noise --prog-scales 6.82 7.34 10.23 13.63 14.68 20.45 --out-tag fairness`; `--sweeps levels --levels 13 15 27 31`; CIFAR: `--prog-scales 1.79 2.13 3.23` | `results/fairness_<ds>.csv` |
| Zusammenfassungen, Befund-Regel, `figures/main_<ds>` | `py -m analog_geometry.analyze --dataset mnist` (ebenso `cifar10`) | `results/sweep_summary*_<ds>.csv`, `findings*_<ds>.csv`, `band_bootstrap_<ds>.csv`, … |
| Abb. 1–3, Gesamttabelle | `py -m analog_geometry.report_figures` | `figures/fig1_drift_variants`, `fig2_noise_levels`, `fig3_delta_w`, `results/findings_all.csv` |
| Alles aus den Roh-CSVs neu | `bash scripts/rebuild_all.sh` (in WSL) | alle Zusammenfassungen und Abbildungen |
| Review-Paket | `bash scripts/build_review.sh` (Git Bash auf dem Host; braucht pandoc und Chrome) | `review/report.pdf`, `fig1–3.png`, `issue9.md` |
| Zahlenprüfung Report | `py scripts/check_report_numbers.py` | `notes/number_check.md` |

## Ordnerstruktur

```
src/analog_geometry/   common.py (Daten, Modelle), train_baseline.py, analog.py (aihwkit-Konfiguration, Varianten),
                       metrics.py (CKA, RSA), sweep.py, isolated.py, analyze.py (Befund-Regel), report_figures.py
scripts/               Smoke-Test, Gate-Prüfungen, Nachtlauf-Treiber, rebuild_all.sh, check_report_numbers.py
tests/                 pytest (Metriken, Augmentierung)
results/               alle CSVs; jede Zahl im Report stammt von hier
figures/               Abbildungen (PNG + PDF)
report/report.md       2-seitiger Bericht
notes/                 step3_design.md (Entscheidungen, Vorab-Regel mit Datum), aihwkit_issues_draft.md (Issue-Entwürfe, nicht gepostet),
                       number_check.md (Zahl -> Quelle)
ENVIRONMENT.md         Versionen, Kompatibilität, Umgehungen
(nicht im Repo: data/, checkpoints/, activations/, logs/)
```

## Reproduzierbarkeit: bekannte Grenze

Alle Läufe mit idealer Peripherie (`forward.is_perfect = True`) sind bei gleichem `noise_seed` bitgenau reproduzierbar. Das Vorwärts- und Peripherie-Rauschen des C++-Simulators nutzt dagegen einen Zufallsgenerator, der sich über die Python-API nicht seeden lässt (er wird aus der Uhrzeit initialisiert, siehe `notes/aihwkit_issues_draft.md` #7). Der einzige Punkt mit aihwkit-Default-Peripherie (`periphery_default`) ist deshalb nicht bitgenau wiederholbar.

## Lizenz

MIT, siehe [LICENSE](LICENSE). Copyright (c) 2026 Nils Leutenegger.
