"""Sanity checks for the analog pipeline on one MNIST checkpoint (no results file; printed only)."""
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import torch

from analog_geometry.analog import AnalogSetting, analog_model
from analog_geometry.common import CKPT_DIR, LAYERS, build_model, collect, eval_loader, get_datasets
from analog_geometry.metrics import linear_cka, rsa

torch.set_num_threads(4)
_, test = get_datasets("mnist")
ev = eval_loader("mnist", test)
dig = build_model("mnist")
dig.load_state_dict(torch.load(CKPT_DIR / "mnist_seed0.pt"))
acc_d, act_d = collect(dig, ev, LAYERS["mnist"])
print(f"digital acc={acc_d:.4f}")

cases = {
    "ideal (prog=0,drift=0,read=0,t=0)": AnalogSetting(prog_scale=0, drift_scale=0, read_scale=0, t_inference=0),
    "cmo default t=1s": AnalogSetting(),
    "cmo prog x4 t=1s": AnalogSetting(prog_scale=4),
    "cmo t=1yr": AnalogSetting(t_inference=3.15e7),
    "cmo k=8, no noise, t=0": AnalogSetting(prog_scale=0, drift_scale=0, read_scale=0, t_inference=0, levels=8),
    "wan2022 t=1s": AnalogSetting(device="wan2022"),
    "cmo default, periphery=default": AnalogSetting(periphery="default"),
}
for name, s in cases.items():
    m = analog_model(dig, s, noise_seed=0)
    acc, act = collect(m, ev, LAYERS["mnist"])
    w = m.fc1.get_weights()[0] if hasattr(m.fc1, "get_weights") else None
    nuniq = len(np.unique(np.round(w.numpy(), 6))) if w is not None else -1
    cka = {k: round(linear_cka(act_d[k], act[k]), 4) for k in act}
    r = {k: round(rsa(act_d[k], act[k]), 4) for k in act}
    print(f"{name:34s} acc={acc:.4f} fc1_unique_w={nuniq} CKA={cka} RSA={r}")
