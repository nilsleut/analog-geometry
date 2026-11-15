"""Shared data, models, and activation capture for analog-geometry."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("AG_DATA_DIR", ROOT / "data"))
RESULTS = ROOT / "results"
CKPT_DIR = ROOT / "checkpoints"
ACT_DIR = ROOT / "activations"

SEEDS = [0, 1, 2, 3, 4]
EVAL_N = 1000
EVAL_SEED = 12345  # fixed; defines the 1000-image evaluation set

# Layers whose outputs are compared (post-nonlinearity for hidden layers, logits for output).
LAYERS = {
    "cifar10": ["act1", "act2", "fc"],
    "mnist": ["act1", "act2", "fc3"],
}

_NORM = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "mnist": ((0.1307,), (0.3081,)),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_datasets(name: str):
    mean, std = _NORM[name]
    norm = transforms.Normalize(mean, std)
    if name == "cifar10":
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), norm,
        ])
        test_tf = transforms.Compose([transforms.ToTensor(), norm])
        ds = datasets.CIFAR10
    elif name == "mnist":
        train_tf = test_tf = transforms.Compose([transforms.ToTensor(), norm])
        ds = datasets.MNIST
    else:
        raise ValueError(name)
    train = ds(DATA_DIR, train=True, download=True, transform=train_tf)
    test = ds(DATA_DIR, train=False, download=True, transform=test_tf)
    return train, test


def as_tensors(ds) -> tuple[torch.Tensor, torch.Tensor]:
    """Whole torchvision dataset -> normalized float tensor [N,C,H,W] + labels (no augmentation)."""
    name = "cifar10" if isinstance(ds, datasets.CIFAR10) else "mnist"
    mean, std = _NORM[name]
    x = torch.as_tensor(np.asarray(ds.data)).float().div_(255.0)
    x = x.permute(0, 3, 1, 2) if x.ndim == 4 else x.unsqueeze(1)
    m = torch.tensor(mean).view(1, -1, 1, 1)
    s = torch.tensor(std).view(1, -1, 1, 1)
    return ((x - m) / s).contiguous(), torch.as_tensor(np.asarray(ds.targets)).long()


def augment_cifar(x: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    """Batched RandomCrop(32, padding=4) + RandomHorizontalFlip (zero padding in normalized space)."""
    n = x.shape[0]
    xp = torch.nn.functional.pad(x, (4, 4, 4, 4))
    dx = torch.randint(0, 9, (n,), generator=g)
    dy = torch.randint(0, 9, (n,), generator=g)
    ar = torch.arange(32)
    rows = (dy[:, None] + ar[None, :])  # [n,32]
    cols = (dx[:, None] + ar[None, :])
    out = xp[torch.arange(n)[:, None, None], :, rows[:, :, None], cols[:, None, :]]  # [n,32,32,C]
    out = out.permute(0, 3, 1, 2)
    flip = torch.rand(n, generator=g) < 0.5
    out[flip] = out[flip].flip(-1)
    return out.contiguous()


def eval_indices(name: str, n_test: int) -> np.ndarray:
    """Fixed 1000-image subset of the test set (same for every run)."""
    rng = np.random.default_rng(EVAL_SEED)
    return np.sort(rng.choice(n_test, size=EVAL_N, replace=False))


def eval_loader(name: str, test, batch_size: int = 500):
    """Fixed eval set as a list of (x, y) batches (deterministic, no augmentation)."""
    idx = torch.as_tensor(eval_indices(name, len(test)))
    x, y = as_tensors(test)
    x, y = x[idx], y[idx]
    return [(x[i:i + batch_size], y[i:i + batch_size]) for i in range(0, len(y), batch_size)]


def full_loader(test, batch_size: int = 1000):
    x, y = as_tensors(test)
    return [(x[i:i + batch_size], y[i:i + batch_size]) for i in range(0, len(y), batch_size)]


class SmallCNN(nn.Module):
    """2 conv + 1 FC for CIFAR-10."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc = nn.Linear(64 * 8 * 8, 10)

    def forward(self, x):
        x = self.pool1(self.act1(self.conv1(x)))
        x = self.pool2(self.act2(self.conv2(x)))
        return self.fc(torch.flatten(x, 1))


class MLP(nn.Module):
    """784-256-256-10 MLP for MNIST."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 256)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.act2(self.fc2(self.act1(self.fc1(x))))
        return self.fc3(x)


def build_model(name: str) -> nn.Module:
    return SmallCNN() if name == "cifar10" else MLP()


@torch.no_grad()
def collect(model: nn.Module, loader: DataLoader, layers: list[str]):
    """Return (accuracy, {layer: [N, D] float32 activations}) on `loader`."""
    model.eval()
    store: dict[str, list[torch.Tensor]] = {k: [] for k in layers}
    mods = dict(model.named_modules())
    hooks = [
        mods[k].register_forward_hook(lambda _m, _i, out, k=k: store[k].append(out.detach().flatten(1).cpu()))
        for k in layers
    ]
    correct = total = 0
    try:
        for x, y in loader:
            pred = model(x).argmax(1)
            correct += int((pred == y).sum())
            total += len(y)
    finally:
        for h in hooks:
            h.remove()
    acts = {k: torch.cat(v).numpy().astype(np.float32) for k, v in store.items()}
    return correct / total, acts


@torch.no_grad()
def accuracy(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        correct += int((model(x).argmax(1) == y).sum())
        total += len(y)
    return correct / total
