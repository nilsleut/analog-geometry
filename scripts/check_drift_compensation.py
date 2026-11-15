"""Check: is GlobalDriftCompensation active in our InferenceRPUConfig, and what does it do to the drift offset?

MNIST, trained seed 0, noise seed 0, CMO, prog_scale 1, ideal periphery.
Writes results/check_drift_compensation.csv.
"""
import csv
import warnings

warnings.filterwarnings("ignore")
import copy

import torch
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.simulator.tiles.inference import InferenceTileWithPeriphery

from analog_geometry.analog import AnalogSetting, make_rpu_config
from analog_geometry.common import CKPT_DIR, LAYERS, RESULTS, build_model, collect, eval_loader, get_datasets
from analog_geometry.metrics import linear_cka, linear_cka_stimcentered, rsa

torch.set_num_threads(8)
_, test = get_datasets("mnist")
ev = eval_loader("mnist", test)
dig = build_model("mnist")
dig.load_state_dict(torch.load(CKPT_DIR / "mnist_seed0.pt"))
_, act_d = collect(dig, ev, LAYERS["mnist"])

rows = []
for t in (86400.0, 365 * 86400.0):
    for drift_mean in (True, False):
        for comp in ("GlobalDriftCompensation", "None"):
            s = AnalogSetting(t_inference=t, drift_mean=drift_mean)
            cfg = make_rpu_config(s)
            if comp == "None":
                cfg.drift_compensation = None
            m = convert_to_analog(copy.deepcopy(dig), cfg).eval()
            torch.manual_seed(0)
            m.program_analog_weights()
            # mean weight before/after drift on fc1 (shows the uniform offset)
            w_prog = m.fc1.get_weights()[0].clone()
            m.drift_analog_weights(t_inference=t)
            w_drift = m.fc1.get_weights()[0]
            alphas = [float(mod.alpha.flatten()[0]) for mod in m.modules()
                      if isinstance(mod, InferenceTileWithPeriphery) and hasattr(mod, "alpha")]
            acc, act = collect(m, ev, LAYERS["mnist"])
            for layer in LAYERS["mnist"]:
                rows.append({"t_inference": t, "drift_mean": drift_mean, "drift_compensation": comp,
                             "alpha_min": min(alphas) if alphas else "", "alpha_max": max(alphas) if alphas else "",
                             "fc1_mean_dw": float((w_drift - w_prog).mean()),
                             "fc1_std_dw": float((w_drift - w_prog).std()),
                             "acc_eval1000": acc, "layer": layer,
                             "cka": linear_cka(act_d[layer], act[layer]),
                             "cka_stimc": linear_cka_stimcentered(act_d[layer], act[layer]),
                             "rsa": rsa(act_d[layer], act[layer])})
            r = rows[-3]
            print(f"t={t:>10.0f} drift_mean={drift_mean!s:5} comp={comp:23} alpha=[{r['alpha_min']:.4g},{r['alpha_max']:.4g}] "
                  if alphas else f"t={t:>10.0f} drift_mean={drift_mean!s:5} comp={comp:23} alpha=- ",
                  f"mean_dw={r['fc1_mean_dw']:+.5f} std_dw={r['fc1_std_dw']:.5f} acc={acc:.3f} "
                  f"CKA={[round(x['cka'], 3) for x in rows[-3:]]} CKAstimc={[round(x['cka_stimc'], 3) for x in rows[-3:]]}")

with (RESULTS / "check_drift_compensation.csv").open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
