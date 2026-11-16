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


## 9. [MAIN DRAFT · QUESTION, not a bug] ReRamCMONoiseModel: single-device converter with drift-free reference turns conductance-independent drift into a uniform weight offset; is this intended?  [code + behaviour] [aihwkit 1.2.0 = master @ 350ff34]
(English, draft only, not posted. Related: #6 converter docstring, #8 missing `g_converter` argument.)

**Question.** `ReRamCMONoiseModel` models the mean conductance relaxation as independent of the programmed conductance (measured, per the docstring). Combined with the hard-wired single-device converter, whose zero reference does not drift, every weight receives the same offset Δw = Δg / scale_ratio. A per-tile scalar drift compensation cannot remove an additive offset. With a differential pair (G+ − G−), the same device model gives no offset (|w| shrinks symmetrically, mostly compensated by `GlobalDriftCompensation`). Is the single-device / drift-free-reference combination intended as the hardware assumption for this model? If so, could the docs say so? If not, should a pair converter (or a drifting reference) be the default or at least selectable?

**Code.**
- `inference/noise/reram.py:222`: `g_converter = SingleDeviceConductanceConverter(g_max=g_max, g_min=g_min)` (no constructor argument, see #8)
- `inference/noise/reram.py:306`: `g_mean = g_prog + (self.decay_dict["mean"][0] * log(t_inference) * self.drift_scale)` (same Δg for every device)
- `inference/converter/conductance.py:380`: `scaled_weights = (weights - w_min) * scale_ratio` (w_min -> g_min, zero sits mid-range; see #6)
- `inference/converter/conductance.py:395`: `weights = params["min"] + ((conductances - self.g_min) / params["scale_ratio"])` (fixed, non-drifting reference)

For the pair variant we override `g_converter` after construction (the constructor has no such argument). We are not sure whether the fitted CMO coefficients are meant to apply per device in a differential pair.

**Minimal reproducer** (run with aihwkit 1.2.0, CPU, 2026-11-16):
```python
import math, torch
from aihwkit.inference.converter.conductance import SinglePairConductanceConverter
from aihwkit.inference.noise.reram import ReRamCMONoiseModel

torch.manual_seed(0)
w = torch.randn(256, 256) * 0.05          # one layer, signed weights
t = 86400.0                               # 1 day
for name, nm in [("single_device (default)", ReRamCMONoiseModel(acceptance_range=2)),
                 ("single_pair (override)", ReRamCMONoiseModel(acceptance_range=2))]:
    if "pair" in name:
        nm.g_converter = SinglePairConductanceConverter(g_max=nm.g_max, g_min=nm.g_min)
    w_prog, nu = nm.apply_programming_noise(w)
    dw = nm.apply_drift_noise(w_prog, nu, t) - w_prog
    m = w.abs().max()
    print(name, {k: f"{100 * dw[mask].mean() / m:+.2f}%" for k, mask in
                 [("w<0", w < -0.01 * m), ("w~0", w.abs() <= 0.01 * m), ("w>0", w > 0.01 * m)]})
nm = ReRamCMONoiseModel()
print("predicted uniform shift:", f"{100 * nm.decay_dict['mean'][0] * math.log(t) / (nm.g_max - nm.g_min) * float(w.max() - w.min()) / float(m):+.2f}% of max|w| (single device)")
```
Output:
```
single_device (default) {'w<0': '-2.52%', 'w~0': '-2.49%', 'w>0': '-2.51%'}
single_pair (override) {'w<0': '+1.33%', 'w~0': '-0.01%', 'w>0': '-1.33%'}
predicted uniform shift: -2.49% of max|w| (single device)
```
Note: the noise model draws from torch's RNG, so with `torch.manual_seed(0)` the output above is identical on repeated runs (5/5 in our runs). Without the seed, the single-device values vary between −2.30 % and −2.54 % and the pair values between ±1.32 % and ±1.36 % of max|w| (5 runs).

**Effect on small networks** (digital training, no HWA training, ideal periphery; accuracy loss vs. the digital net, mean ± SD over 5 seeds × 3 programming runs):

<!-- When posting: upload figures/fig0_headline.png (drag & drop into the GitHub editor) and replace the placeholder path below. -->
![Accuracy loss vs. drift time for single device / fixed reference, single device / drifting reference, and differential pair](PLACEHOLDER/fig0_headline.png)

| Drift | MNIST MLP, single device | MNIST MLP, pair | CIFAR-10 CNN, single device | CIFAR-10 CNN, pair |
|---|---|---|---|---|
| 1 day | 0.49 ± 0.16 pp | 0.00 ± 0.05 pp | 30.0 ± 3.5 pp | 1.1 ± 0.4 pp |
| 1 week | 0.85 ± 0.39 pp | 0.02 ± 0.05 pp | 35.9 ± 3.1 pp | 1.4 ± 0.5 pp |
| 1 year | 2.54 ± 1.32 pp | 0.02 ± 0.07 pp | 45.1 ± 1.8 pp | 2.2 ± 0.8 pp |

These are small networks (784-256-256-10 MLP, 2-conv CNN) trained digitally without hardware-aware training, so the absolute numbers are not representative of deployed models; we report them only to show that the converter choice dominates the drift effect under otherwise identical settings.

Setting only the mean drift term to 0 (≈ a reference that drifts along) gives 2.2 ± 0.5 pp on CIFAR at 1 day.

<details><summary>Side observation: converter choice also affects programming noise</summary>

With the single-device converter, weights near zero sit mid-range, where the conductance-dependent CMO programming noise (σ = 0.0112 + 0.0113·g µS, acceptance 2 %) is largest. Per weight, the programming noise is therefore 2.4–4.9× larger than with a differential pair (per tile, in units of max|w|). Once the effective resolution is matched (noise scale with equal per-weight error; levels with equal weight step), the accuracy difference between the two converters disappears in our runs. So the converter choice also matters for noise comparisons, not only for drift. This is an observation, not a bug report.

</details>

**Source questions.**
1. The docstring cites "Falcone et al. (In Review)" (reram.py:172-173) without a title. Is this arXiv:2502.04524 (Falcone et al., "All-in-One Analog AI Hardware: On-Chip Training and Inference with Conductive-Metal-Oxide/HfOx ReRAM Devices")?
2. If so, was the measured array read out with a single device per weight against a fixed reference, or as a differential pair? The paper says that "pre-trained normalized weights are initially mapped into target conductances" (Sec. 2.2.1), but we could not find how signed weights or the zero reference were realised.
3. In that paper the relaxation is characterised up to 1 week (Sec. 2.2.2). The model accepts any `t_inference` and extrapolates the ln(t) fit. Is extrapolation beyond 1 week intended (e.g. the 1-year values in the table above)?

