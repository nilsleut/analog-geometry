# Environment & aihwkit compatibility (Schritt 1, 2026-11-14)

## Ergebnis
aihwkit funktioniert -> **kein Fallback-Modul nötig**. Alle Läufe in WSL2.

| Komponente | Version |
|---|---|
| Host | Windows 11 Home 10.0.26200, keine NVIDIA-GPU (CPU only, 14 Threads, 15 GB RAM in WSL) |
| Laufzeit | WSL2 Ubuntu 24.04, Kernel 6.6.87.2-microsoft-standard-WSL2, glibc 2.39 |
| Python | 3.12.3 (venv unter `~/.venvs/analog-geometry` in WSL) |
| torch / torchvision | 2.15.0+cpu / 0.29.2+cpu |
| aihwkit | 1.2.0 (PyPI manylinux wheel) |

Smoke-Test: `scripts/smoke_test_aihwkit.py` -> `results/env_smoke_test.csv`.

## Was nicht ging
- **Natives Windows (Python 3.11, uv):** Für aihwkit 1.2.0 gibt es auf PyPI kein Windows-Wheel, deshalb wird ein Quell-Build
  versucht. MSVC 19.51 ist vorhanden, aber CMake bricht ab: `Could not find OpenBLAS` (cmake/Modules/FindOpenBLAS.cmake:79).
  Nicht weiter verfolgt, weil WSL funktioniert.

## Bezug zu IBM/aihwkit#828
#828: aihwkit 1.1.0 + torch >= 2.13 -> `AnalogLinear` mit `InferenceRPUConfig` scheitert ("weights must be a CPU tensor"),
mit `SingleRPUConfig` "Invalid weights dimensions". Mit **aihwkit 1.2.0 + torch 2.15.0+cpu** laufen im Smoke-Test
`InferenceRPUConfig`, `SingleRPUConfig` und `TorchInferenceRPUConfig` (Linear und Conv2d, Forward) ohne Fehler.
Der Fix aus #828 ist für diese Kombination damit bestätigt.

## Befunde mit Folgen für Schritt 3
- `ReRamWan2022NoiseModel` (aihwkit.inference.noise.reram) akzeptiert in `drift_analog_weights` nur
  `t_inference` in {1, 86400, 172800} s, sonst `ArgumentError`. Mit diesem Preset hat ein Drift-Sweep also höchstens 3 Punkte.
- `ReRamWan2022NoiseModel` hat nur ein globales `noise_scale`, kein separates `prog_noise_scale` / `drift_scale`
  (anders als `PCMLikeNoiseModel`).
- Für Stufenquantisierung gibt es `WeightModifierType.DISCRETIZE` (Feld `res`). Ob das beim Programmieren für Inferenz greift
  oder nur als Trainingsmodifikator, ist noch offen und wird in Schritt 3 geprüft.

## Umgehungen von aihwkit-Bugs (im eigenen Code, installiertes Paket unverändert)
Geprüft gegen IBM/aihwkit `master` @ 350ff34 (Commit vom 2026-11-05; der Default-Branch heißt `master`, nicht `main`).
`inference/noise/reram.py` ist dort byte-identisch mit dem installierten 1.2.0 (gleicher md5), beide Bugs bestehen also weiter.
- **ReRamCMONoiseModel `acceptance_range`** (Entwurf #5): Der Default `2e-2` ist kein Schlüssel von `coeff_dict` {0.2, 2} und fällt still
  auf 0.2 (%) zurück. Projektentscheidung: 2 % Akzeptanzbereich. Umgehung: `src/analog_geometry/analog.py` übergibt
  `acceptance_range=2` explizit und bricht mit Fehler ab, falls der Wert nicht übernommen wird.
- **ReRamWan2022NoiseModel `coeff_g_max_reference`** (Entwurf #3): Ein explizit übergebener Wert führt zu AttributeError. Umgehung: Der
  Parameter wird nicht übergeben (Default-Verhalten, Skalierung g_max/g_max = 1).
Details und Reproducer: `notes/aihwkit_issues_draft.md`.

## Reproduktion
```bash
# in WSL Ubuntu 24.04
python3 -m venv ~/.venvs/analog-geometry
~/.venvs/analog-geometry/bin/pip install -r requirements.txt
~/.venvs/analog-geometry/bin/python scripts/smoke_test_aihwkit.py
```
