# aihwkit: Probleme und Doku-Lücken (NUR ENTWÜRFE, nichts gepostet)

Upstream-Prüfung: IBM/aihwkit `master` @ 350ff34 (2026-11-05); reram.py byte-identisch zu 1.2.0 (md5 cdfee337...).

Status-Legende: [beobachtet] = reproduziert; [Doku prüfen] = vor einem Issue klären, ob es schon dokumentiert ist / ob es ein Duplikat gibt.

## 1. Windows: Quell-Build bricht ohne Hinweis auf OpenBLAS ab  [beobachtet] [Doku prüfen]
Umgebung: Windows 11, CPython 3.11.9, MSVC 19.51, aihwkit 1.2.0 sdist (kein Windows-Wheel auf PyPI).
Reproducer:
    uv venv --python 3.11 .venv
    uv pip install --python .venv/Scripts/python.exe aihwkit
Ist: CMake Error at cmake/Modules/FindOpenBLAS.cmake:79: Could not find OpenBLAS
Wunsch: klarer Hinweis in der Install-Doku / Fehlermeldung (Windows nicht unterstützt bzw. OpenBLAS-Pfad setzen, WSL empfehlen).

## 2. ReRamWan2022NoiseModel: nur 3 diskrete Drift-Zeiten  [beobachtet] [IST DOKUMENTIERT, kein Bug, kein Issue nötig]
Nachtrag: Der Klassen-Docstring nennt "only 1 sec, 1 day, 2 day", der Docstring von apply_drift_noise_to_conductance das ArgumentError.
Übrig bleibt höchstens ein kleiner Wunsch: schon bei der Konfiguration validieren statt erst beim Drift. Niedrige Priorität.
Umgebung: WSL2 Ubuntu 24.04, Python 3.12.3, torch 2.15.0+cpu, aihwkit 1.2.0.
Reproducer:
    import torch
    from aihwkit.nn import AnalogLinear
    from aihwkit.simulator.configs import InferenceRPUConfig
    from aihwkit.inference.noise.reram import ReRamWan2022NoiseModel
    cfg = InferenceRPUConfig(); cfg.noise_model = ReRamWan2022NoiseModel()
    m = AnalogLinear(16, 8, rpu_config=cfg).eval()
    m.program_analog_weights()
    m.drift_analog_weights(t_inference=3600.0)
Ist: ArgumentError: t_inference should be one of `[1.0, 86400.0, 172800.0]` (erst beim Drift, nicht schon bei der Konfiguration).
Fragen/Wunsch: Einschränkung im Docstring nennen; Interpolation anbieten oder früher validieren.

## 3. ReRamWan2022NoiseModel: explizites `coeff_g_max_reference` -> AttributeError  [beobachtet] [besteht auf master @ 350ff34]
Umgebung: wie 2.
Reproducer:
    import torch
    from aihwkit.inference.noise.reram import ReRamWan2022NoiseModel
    nm = ReRamWan2022NoiseModel(coeff_g_max_reference=40.0)
    nm.apply_programming_noise_to_conductance(torch.full((4,), 10.0))
Ist: AttributeError: 'ReRamWan2022NoiseModel' object has no attribute 'coeff_g_max_reference'
Ursache (reram.py, __init__): `self.coeff_g_max_reference` wird nur im Zweig `if coeff_g_max_reference is None` gesetzt, ein else-Zweig fehlt.
Verwandt (Frage, kein sicherer Bug): Bei den Default-Koeffizienten wird `self.prog_coeff_g_max_reference = 40.0` gesetzt, aber nirgends
gelesen. `_apply_poly` nutzt `coeff_g_max_reference` (= g_max = 25 µS), also Skalierung 1 statt 25/40. Ist das beabsichtigt?

## 4. Kein eingebauter Weg, endlich viele Leitwertstufen beim Programmieren (Inferenz) zu modellieren  [Doku prüfen]
Gefunden habe ich nur `WeightModifierType.DISCRETIZE` (Modifikator im Training) und `forward.inp_res/out_res` (DAC/ADC).
Für das Projekt wird ein Noise-Model-Wrapper genutzt, der g_target vor dem Programmierrauschen auf k Stufen rundet.
Wenn es dafür einen offiziellen Weg gibt, fehlt er in der Doku; sonst wäre das ein Feature-Wunsch.

## 5. ReRamCMONoiseModel: Default `acceptance_range=2e-2` passt nicht zu den Dict-Schlüsseln {0.2, 2} -> stiller Fallback auf 0.2  [aus Code + Verhalten] [besteht auf master @ 350ff34]
Umgebung: wie 2.
Reproducer:
    from aihwkit.inference.noise.reram import ReRamCMONoiseModel
    print(ReRamCMONoiseModel().acceptance_range)   # 0.2, nicht 2e-2 bzw. 2
Code: `if acceptance_range not in coeff_dict.keys(): acceptance_range = min(coeff_dict.keys())`.
Die Schlüssel sind offenbar Prozent (0.2 %, 2 %), der Default ist ein Bruch (2e-2 = 2 %). Der Fallback wählt still die engste Toleranz.
Folge im Projekt: Das Default-Programmierrauschen ist fast null (MNIST-MLP, CKA >= 0.9997 selbst bei prog_noise_scale=4).
Wunsch: Einheit klären, bei unbekanntem Wert warnen oder Fehler werfen.

## 6. SingleDeviceConductanceConverter: Docstring „g_min = logische Null“ widerspricht dem Code (g_min = w_min)  [aus Code] [besteht auf master @ 350ff34]
Docstring: "g_min: ... the minimal conductance, ie the value the logical zero of the weights will be mapped to."
Code (convert_to_conductances): `scale_ratio = (g_max - g_min) / (w_max - w_min)`, `conductance = (weights - w_min) * scale_ratio + g_min`.
Auf g_min wird also das kleinste Gewicht abgebildet, nicht die Null. Bei vorzeichenbehafteten Gewichten liegt die Null in der Mitte des Bereichs.
Folge: Eine gleichförmige Leitwertverschiebung (z. B. Drift-Mittelwert in ReRamCMONoiseModel) wird zu einem gleichförmigen Gewichtsversatz.
Diesen kann GlobalDriftCompensation (ein skalarer Faktor) prinzipiell nicht korrigieren. Das ist kein Bug, sollte aber dokumentiert sein.
Reproducer:
    import torch
    from aihwkit.inference.converter.conductance import SingleDeviceConductanceConverter
    c = SingleDeviceConductanceConverter()
    g, p = c.convert_to_conductances(torch.tensor([-1.0, 0.0, 1.0]))
    print(g)   # tensor([ 9.0000, 48.6000, 88.2000]) -> die Null liegt bei 48.6, nicht bei g_min=9
Nachtrag 2026-11-16 (Gate): Der Docstring ist offenbar von SinglePairConductanceConverter übernommen. Dort stimmt „logische Null -> g_min“,
und eine gleichförmige Leitwertverschiebung hebt sich in G+ - G- auf. Der Widerspruch ist also nicht die Ursache des Versatzes, beschreibt aber
genau das Verhalten (Differenz zweier Geräte), das der Konverter NICHT hat. Gemessen (results/gate_drift_offset.csv, Drift 1 d):
Δw_drift ist für positive, negative und ~0-Gewichte gleich und entspricht −0,089·ln(t)/scale_ratio auf < 2 %.

## 7. Kein (gefundener) Weg, den RNG des C++-Simulators für Forward-/Peripherie-Rauschen zu seeden  [beobachtet] [Doku/Quelle geprüft 2026-11-15, master @ 350ff34]
Umgebung: wie 2. Mit `torch.manual_seed` vor program/drift sind alle Läufe mit `forward.is_perfect=True` bitgenau reproduzierbar.
Mit den Default-IOParameters (w_noise/out_noise aktiv) weichen zwei identische Läufe ab (MNIST-MLP: max |ΔCKA| 1.6e-3, |ΔAcc| 0.21 pp).
In `aihwkit.simulator.rpu_base` habe ich keine Methode mit "seed" im Namen gefunden (dir() von rpu_base, rpu_base.tiles, FloatingPointTile).
Prüfung (2026-11-15):
- Doku-Seite https://aihwkit.readthedocs.io/en/latest/using_simulator.html: keine Erwähnung von Seed, RNG oder Reproduzierbarkeit
  (nur diese Seite geprüft, nicht die ganze Doku).
- Quelle master @ 350ff34, src/rpucuda/rng.cpp: `RealWorldRNG<T>::setSeed(seed)` nimmt bei seed == 0 die Uhrzeit
  (`std::chrono::high_resolution_clock::now()`). Das passt zur beobachteten Nicht-Reproduzierbarkeit.
- Python-Bindings (src/aihwkit/simulator/rpu_base_src/rpu_base*.cpp): Einziger Treffer für "seed" ist `construction_seed`
  in rpu_base_devices.cpp (Geräteparameter beim Konstruieren), kein Seed für das Forward-Rauschen der Tiles.
  Mit welchem Seed die Tile-RNG konstruiert wird, habe ich nicht im Tile-Code verfolgt.
Vorschlag fürs Issue: Seed für das Forward-/IO-Rauschen pro Tile oder global exponieren, oder dokumentieren, dass es zeitbasiert ist.
Möglicher Workaround (nicht getestet): TorchInferenceRPUConfig, falls dessen Rauschen über torch-RNG läuft.

## 8. ReRamCMONoiseModel: Docstring nennt Argument `g_converter`, `__init__` nimmt es nicht an (Konverter fest verdrahtet)  [aus Code + Verhalten] [besteht auf master @ 350ff34]
Docstring (reram.py:196): "g_converter: Instantiated class of the conductance converter for a single device per cross-point."
Code (reram.py:210-222): keine Option `g_converter` in der Signatur; `g_converter = SingleDeviceConductanceConverter(g_max=g_max, g_min=g_min)` ist fest.
Folge: Ein Differenzpaar-Konverter lässt sich nur durch Überschreiben des Attributs nach der Konstruktion setzen (so in analog.py, Paket nicht gepatcht).
Ob das Modell mit einem Paar-Konverter physikalisch gültig ist (die Drift-Fits stammen aus Einzelgeräten), sagt die Doku nicht.
Reproducer (geprüft 2026-11-16, aihwkit 1.2.0):
    from aihwkit.inference.converter.conductance import SinglePairConductanceConverter
    from aihwkit.inference.noise.reram import ReRamCMONoiseModel
    print(type(ReRamCMONoiseModel().g_converter).__name__)          # SingleDeviceConductanceConverter
    ReRamCMONoiseModel(g_converter=SinglePairConductanceConverter())  # TypeError: unexpected keyword argument 'g_converter'

