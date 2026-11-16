"""Analog (aihwkit) versions of the digitally trained networks.

Inference only: digital weights -> convert_to_analog -> program -> drift -> evaluate.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
from aihwkit.inference.converter.conductance import SinglePairConductanceConverter
from aihwkit.inference.noise.reram import ReRamCMONoiseModel, ReRamWan2022NoiseModel
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.simulator.configs import InferenceRPUConfig


def _quantize(g: torch.Tensor, levels: int | None, g_min: float, g_max: float) -> torch.Tensor:
    """Round conductances to `levels` equally spaced values in [g_min, g_max]."""
    if levels is None:
        return g
    step = (g_max - g_min) / (levels - 1)
    return (torch.round((g.clamp(g_min, g_max) - g_min) / step) * step + g_min)


class QuantizedCMO(ReRamCMONoiseModel):
    """ReRamCMONoiseModel + target conductances snapped to k levels before programming noise (own extension)."""

    def __init__(self, levels: int | None = None, **kw):
        super().__init__(**kw)
        self.levels = levels

    def apply_programming_noise_to_conductance(self, g_target):
        return super().apply_programming_noise_to_conductance(
            _quantize(g_target, self.levels, self.g_min, self.g_max))


class QuantizedWan2022(ReRamWan2022NoiseModel):
    """ReRamWan2022NoiseModel + k levels in [0, g_max]. Drift resamples from g_target, so quantize there too."""

    def __init__(self, levels: int | None = None, **kw):
        super().__init__(**kw)
        self.levels = levels

    def apply_programming_noise_to_conductance(self, g_target):
        return super().apply_programming_noise_to_conductance(_quantize(g_target, self.levels, 0.0, self.g_max))

    def generate_drift_coefficients(self, g_target):
        return _quantize(g_target, self.levels, 0.0, self.g_max)


@dataclass(frozen=True)
class AnalogSetting:
    device: str = "cmo"            # "cmo" | "wan2022"
    prog_scale: float = 1.0        # cmo: prog_noise_scale; wan2022: noise_scale (prog + drift together)
    drift_scale: float = 1.0       # cmo only
    read_scale: float = 1.0        # cmo only
    # cmo only: program-verify acceptance range in PERCENT, must be a key of coeff_dict (0.2 | 2).
    # Workaround for aihwkit 1.2.0 / master@350ff34: the library default 2e-2 is not a key and silently
    # falls back to 0.2. We pass 2 explicitly (decision: 2 %). See notes/aihwkit_issues_draft.md #5.
    acceptance: float = 2.0
    drift_mean: bool = True        # cmo only: False = drift without mean shift (offset control, decay_dict mean := 0)
    levels: int | None = None      # None = continuous conductances
    t_inference: float = 1.0       # seconds after programming; 0 = no drift step
    periphery: str = "ideal"       # "ideal" (perfect forward pass) | "default" (aihwkit InferenceRPUConfig defaults)

    def as_row(self) -> dict:
        d = asdict(self)
        d["levels"] = "inf" if self.levels is None else self.levels
        if self.device != "cmo":  # parameters without effect for wan2022
            d.update(drift_scale="", read_scale="", acceptance="", drift_mean="")
        return d


CONVERTERS = ("single_device", "single_pair")


def make_rpu_config(s: AnalogSetting, converter: str = "single_device") -> InferenceRPUConfig:
    """`converter` (cmo only): "single_device" = aihwkit default (one device per weight, w_min -> g_min);
    "single_pair" = differential pair G+ - G- (SinglePairConductanceConverter), same g range. ReRamCMONoiseModel
    hard-codes SingleDeviceConductanceConverter in __init__ (no g_converter argument in 1.2.0 / master@350ff34),
    so the pair converter is set on the instance afterwards (own code, package not patched). Gate 2026-11-16."""
    if converter not in CONVERTERS:
        raise ValueError(converter)
    cfg = InferenceRPUConfig()
    if s.device == "cmo":
        extra = {}
        if not s.drift_mean:
            decay = copy.deepcopy(ReRamCMONoiseModel().decay_dict)  # library defaults, only the mean term zeroed
            decay["mean"] = [0.0 for _ in decay["mean"]]
            extra["decay_dict"] = decay
        nm = QuantizedCMO(levels=s.levels, prog_noise_scale=s.prog_scale,
                          drift_scale=s.drift_scale, read_noise_scale=s.read_scale,
                          acceptance_range=s.acceptance, **extra)
        if nm.acceptance_range != s.acceptance:  # guard against the silent fallback
            raise ValueError(f"acceptance {s.acceptance} not in {list(nm.coeff_dict)}")
        if converter == "single_pair":
            nm.g_converter = SinglePairConductanceConverter(g_max=nm.g_max, g_min=nm.g_min)
        cfg.noise_model = nm
    elif s.device == "wan2022":
        if converter != "single_device":
            raise ValueError("converter option is for cmo only")
        cfg.noise_model = QuantizedWan2022(levels=s.levels, noise_scale=s.prog_scale)
    else:
        raise ValueError(s.device)
    if s.periphery == "ideal":
        cfg.forward.is_perfect = True
    elif s.periphery != "default":
        raise ValueError(s.periphery)
    return cfg


def analog_model(digital: nn.Module, s: AnalogSetting, noise_seed: int, converter: str = "single_device") -> nn.Module:
    """Convert, program and drift. `noise_seed` fixes the device-noise draw."""
    m = convert_to_analog(copy.deepcopy(digital), make_rpu_config(s, converter))
    m.eval()
    torch.manual_seed(noise_seed)
    m.program_analog_weights()
    if s.t_inference > 0:
        m.drift_analog_weights(t_inference=s.t_inference)
    return m
