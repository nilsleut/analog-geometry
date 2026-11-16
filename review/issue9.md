# ReRamCMONoiseModel: single-device converter with drift-free reference turns conductance-independent drift into a uniform weight offset. Is this intended?

_Question, not a bug report. aihwkit 1.2.0 (= master @ 350ff34), CPU._

**Question.** `ReRamCMONoiseModel` models the mean conductance relaxation as independent of the programmed conductance (measured, per the docstring). Combined with the hard-wired single-device converter, whose zero reference does not drift, every weight receives the same offset Δw = Δg / scale_ratio. A per-tile scalar drift compensation cannot remove an additive offset. With a differential pair (G+ − G−), the same device model gives no offset (|w| shrinks symmetrically, mostly compensated by `GlobalDriftCompensation`). Is the single-device / drift-free-reference combination intended as the hardware assumption for this model? If so, could the docs say so? If not, should a pair converter (or a drifting reference) be the default or at least selectable?

**Code.**
- `inference/noise/reram.py:222`: `g_converter = SingleDeviceConductanceConverter(g_max=g_max, g_min=g_min)` (no constructor argument)
- `inference/noise/reram.py:306`: `g_mean = g_prog + (self.decay_dict["mean"][0] * log(t_inference) * self.drift_scale)` (same Δg for every device)
- `inference/converter/conductance.py:380`: `scaled_weights = (weights - w_min) * scale_ratio` (w_min -> g_min, zero sits mid-range)
- `inference/converter/conductance.py:395`: `weights = params["min"] + ((conductances - self.g_min) / params["scale_ratio"])` (fixed, non-drifting reference)

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

**Effect on small networks** (digital training, no HWA training, ideal periphery, 5 seeds × 3 programming runs; accuracy loss vs. the digital net):

| Drift | MNIST MLP, single device | MNIST MLP, pair | CIFAR-10 CNN, single device | CIFAR-10 CNN, pair |
|---|---|---|---|---|
| 1 day | 0.49 pp | 0.00 pp | 30.0 pp | 1.1 pp |
| 1 week | 0.85 pp | 0.02 pp | 35.9 pp | 1.4 pp |
| 1 year | 2.54 pp | 0.02 pp | 45.1 pp | 2.2 pp |

Setting only the mean drift term to 0 (≈ a reference that drifts along) gives 2.2 pp on CIFAR at 1 day.

**Side observation (not a bug).** With the single-device converter, weights near zero sit mid-range, where the conductance-dependent CMO programming noise (σ = 0.0112 + 0.0113·g µS, acceptance 2 %) is largest. Per weight, the programming noise is therefore 2.4–4.9× larger than with a differential pair (per tile, in units of max|w|). Once the effective resolution is matched (noise scale with equal per-weight error; levels with equal weight step), the accuracy difference between the two converters disappears in our runs. So the converter choice also matters for noise comparisons, not only for drift.

**Source questions.**
1. The docstring cites "Falcone et al. (In Review)" (reram.py:172-173) without a title. Is this arXiv:2502.04524 (Falcone et al., "All-in-One Analog AI Hardware: On-Chip Training and Inference with Conductive-Metal-Oxide/HfOx ReRAM Devices")?
2. If so, was the measured array read out with a single device per weight against a fixed reference, or as a differential pair? The paper says that "pre-trained normalized weights are initially mapped into target conductances" (Sec. 2.2.1), but we could not find how signed weights or the zero reference were realised.
3. In that paper the relaxation is characterised up to 1 week (Sec. 2.2.2). The model accepts any `t_inference` and extrapolates the ln(t) fit. Is extrapolation beyond 1 week intended (e.g. the 1-year values in the table above)?

