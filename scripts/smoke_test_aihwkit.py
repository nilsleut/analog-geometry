"""Smoke test: which aihwkit configs work with the installed torch version?

Records versions and pass/fail per config to results/env_smoke_test.csv.
Reference: IBM/aihwkit#828 (aihwkit 1.1.0 + torch>=2.13 broke InferenceRPUConfig).
"""
import csv
import platform
import sys
import traceback
from pathlib import Path

import torch
import aihwkit
from aihwkit.nn import AnalogLinear, AnalogConv2d
from aihwkit.simulator.configs import (
    InferenceRPUConfig,
    SingleRPUConfig,
    TorchInferenceRPUConfig,
)

OUT = Path(__file__).resolve().parents[1] / "results" / "env_smoke_test.csv"


def try_case(name, fn):
    try:
        info = fn()
        return {"case": name, "ok": True, "info": info or ""}
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc().strip().splitlines()[-1]
        return {"case": name, "ok": False, "info": f"{type(e).__name__}: {e} | {tb}"[:300]}


def linear_fwd(cfg):
    def f():
        m = AnalogLinear(4, 2, rpu_config=cfg)
        y = m(torch.randn(3, 4))
        return f"out={tuple(y.shape)}"
    return f


def conv_fwd(cfg):
    def f():
        m = AnalogConv2d(3, 8, 3, rpu_config=cfg)
        y = m(torch.randn(2, 3, 8, 8))
        return f"out={tuple(y.shape)}"
    return f


def reram_noise_model():
    from aihwkit.inference import noise as nz
    names = [n for n in dir(nz) if "ReRam" in n or "RRAM" in n.upper()]
    return "available: " + ",".join(names)


def program_and_drift(cfg_factory):
    def f():
        cfg = cfg_factory()
        m = AnalogLinear(16, 8, rpu_config=cfg)
        m.eval()
        x = torch.randn(5, 16)
        m.program_analog_weights()
        y0 = m(x)
        m.drift_analog_weights(t_inference=3600.0)
        y1 = m(x)
        return f"|y_drift - y_prog|={float((y1 - y0).abs().mean()):.4f}"
    return f


def reram_cfg():
    from aihwkit.inference.noise.reram import ReRamWan2022NoiseModel
    cfg = InferenceRPUConfig()
    cfg.noise_model = ReRamWan2022NoiseModel()
    return cfg


def main():
    rows = [
        try_case("Linear/InferenceRPUConfig", linear_fwd(InferenceRPUConfig())),
        try_case("Linear/SingleRPUConfig", linear_fwd(SingleRPUConfig())),
        try_case("Linear/TorchInferenceRPUConfig", linear_fwd(TorchInferenceRPUConfig())),
        try_case("Conv2d/InferenceRPUConfig", conv_fwd(InferenceRPUConfig())),
        try_case("Conv2d/TorchInferenceRPUConfig", conv_fwd(TorchInferenceRPUConfig())),
        try_case("ReRAM noise models in aihwkit.inference.noise", reram_noise_model),
        try_case("program+drift/InferenceRPUConfig(PCM default)", program_and_drift(InferenceRPUConfig)),
        try_case("program+drift/InferenceRPUConfig(ReRamWan2022)", program_and_drift(reram_cfg)),
    ]
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "aihwkit": aihwkit.__version__,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(env) + ["case", "ok", "info"])
        w.writeheader()
        for r in rows:
            w.writerow({**env, **r})
    print(env)
    for r in rows:
        print(("PASS " if r["ok"] else "FAIL ") + r["case"] + " :: " + r["info"])


if __name__ == "__main__":
    main()
