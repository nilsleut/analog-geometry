"""Schritt 2: digital baselines (backprop), 5 seeds per dataset.

Writes
  results/baseline_accuracy.csv        one row per (dataset, seed)
  results/eval_indices_<dataset>.csv   the fixed 1000-image evaluation subset
  checkpoints/<dataset>_seed<s>.pt
  activations/digital_<dataset>_seed<s>.npz   per-layer activations on the eval set

Usage: PYTHONPATH=src python -m analog_geometry.train_baseline --dataset cifar10 [--seeds 0 1 2 3 4]
"""
from __future__ import annotations

import argparse
import csv
import time

import numpy as np
import torch
import torch.nn as nn
from analog_geometry.common import (
    ACT_DIR, CKPT_DIR, LAYERS, RESULTS, SEEDS, accuracy, as_tensors, augment_cifar,
    build_model, collect, eval_indices, eval_loader, full_loader, get_datasets, set_seed,
)

HP = {
    "cifar10": dict(epochs=20, lr=1e-3, batch_size=128, weight_decay=5e-4),
    "mnist": dict(epochs=10, lr=1e-3, batch_size=128, weight_decay=0.0),
}
FIELDS = ["dataset", "seed", "epochs", "lr", "batch_size", "weight_decay",
          "train_loss_last_epoch", "test_acc_full", "test_acc_eval1000", "train_seconds"]


def train_one(name: str, seed: int, train, test) -> dict:
    hp = HP[name]
    set_seed(seed)
    g = torch.Generator().manual_seed(seed)
    xtr, ytr = train
    bs = hp["batch_size"]
    model = build_model(name)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=hp["epochs"])
    loss_fn = nn.CrossEntropyLoss()
    t0 = time.time()
    for ep in range(hp["epochs"]):
        model.train()
        tot, n = 0.0, 0
        perm = torch.randperm(len(ytr), generator=g)
        xe = augment_cifar(xtr[perm], g) if name == "cifar10" else xtr[perm]
        ye = ytr[perm]
        for i in range(0, len(ye), bs):
            x, y = xe[i:i + bs], ye[i:i + bs]
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            tot += float(loss) * len(y)
            n += len(y)
        sched.step()
        print(f"[{name} seed{seed}] epoch {ep + 1}/{hp['epochs']} loss {tot / n:.4f}", flush=True)
    secs = time.time() - t0

    acc_full = accuracy(model, full_loader(test))
    acc_eval, acts = collect(model, eval_loader(name, test), LAYERS[name])
    CKPT_DIR.mkdir(exist_ok=True)
    ACT_DIR.mkdir(exist_ok=True)
    torch.save(model.state_dict(), CKPT_DIR / f"{name}_seed{seed}.pt")
    np.savez_compressed(ACT_DIR / f"digital_{name}_seed{seed}.npz", **acts)
    return {"dataset": name, "seed": seed, **hp, "train_loss_last_epoch": round(tot / n, 5),
            "test_acc_full": acc_full, "test_acc_eval1000": acc_eval, "train_seconds": round(secs, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cifar10", "mnist"], required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = ap.parse_args()
    torch.set_num_threads(12)

    train, test = get_datasets(args.dataset)
    RESULTS.mkdir(exist_ok=True)
    idx = eval_indices(args.dataset, len(test))
    with (RESULTS / f"eval_indices_{args.dataset}.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["test_index", "label"])
        for i in idx:
            w.writerow([int(i), int(test.targets[i])])

    out = RESULTS / "baseline_accuracy.csv"
    existing = []
    if out.exists():
        with out.open() as fh:
            existing = [r for r in csv.DictReader(fh)
                        if not (r["dataset"] == args.dataset and int(r["seed"]) in args.seeds)]
    train_t = as_tensors(train)
    rows = existing + [train_one(args.dataset, s, train_t, test) for s in args.seeds]
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
