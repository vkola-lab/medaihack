"""
Training script for Amyloid PET Centiloid Prediction.

Entry point:
    python dev/train.py --config dev/config/train.yaml

All hyperparameters, paths, augmentation and loss settings live in the YAML.
To run a variant experiment, copy the config file, edit the fields you care
about, and point --config at the new file — the YAML is the full run record.

EDA-informed defaults:
  • NAV (n=85) and PIB get STRONG augmentation when augmentation.per_tracer=true
  • model.freeze_tracer_norm=true runs the TracerNorm ablation (γ=1, β=0, frozen)
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mygo_centiloid import (
    PETDataset,
    PETResNet, PETResNetNoGAP, PETResNetNoFiLM, PETResNetTracerNormOnly,
    get_criterion,
)
from mygo_centiloid.utils import (
    make_run_dir, write_config, write_metrics, append_epoch, append_registry,
)

try:
    from mygo_centiloid import build_train_transform
except ImportError:
    build_train_transform = None


# ─────────────────────────────────────────────────────────────────────────────
# Config loading
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    # Minimal validation — fail fast with a useful message
    required = {
        "paths":        ["train_csv", "val_csv", "log_dir", "checkpoint_dir"],
        "run":          ["name", "seed"],
        "model":        ["name", "emb_dim", "dropout_high", "dropout_low",
                         "freeze_tracer_norm"],
        "training":     ["epochs", "batch_size", "num_workers", "lr",
                         "weight_decay", "clip_grad", "patience"],
        "augmentation": ["enabled", "per_tracer"],
        "loss":         ["name", "delta", "alpha"],
    }
    for section, keys in required.items():
        if section not in cfg:
            raise KeyError(f"Config missing section: [{section}]")
        for k in keys:
            if k not in cfg[section]:
                raise KeyError(f"Config missing key: [{section}].{k}")
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Weighted Sampler
# ─────────────────────────────────────────────────────────────────────────────

def build_weighted_sampler(centiloids: np.ndarray) -> WeightedRandomSampler:
    """Inverse-frequency sampler across 6 CL bins — fixes 64.8 % negative bias."""
    bin_edges = [-np.inf, 0, 25, 50, 75, 100, np.inf]
    bin_idx   = np.digitize(centiloids, bin_edges) - 1
    bin_idx   = np.clip(bin_idx, 0, len(bin_edges) - 2)

    counts  = np.bincount(bin_idx, minlength=len(bin_edges) - 1).astype(float)
    counts  = np.where(counts == 0, 1.0, counts)

    weights = 1.0 / counts[bin_idx]
    weights = weights / weights.sum() * len(weights)

    return WeightedRandomSampler(
        weights=torch.FloatTensor(weights),
        num_samples=len(centiloids),
        replacement=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def pearson_r(pred: torch.Tensor, target: torch.Tensor) -> float:
    vx = pred   - pred.mean()
    vy = target - target.mean()
    return ((vx * vy).sum() / (vx.norm() * vy.norm() + 1e-8)).item()


# ─────────────────────────────────────────────────────────────────────────────
# Train / Val loops
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, scaler, device, clip_grad):
    model.train()
    running_loss = 0.0

    for images, targets, tracers in loader:
        images  = images.to(device,  non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        tracers = tracers.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device.type, enabled=scaler.is_enabled()):
            preds = model(images, tracers)
            loss  = criterion(preds, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * len(images)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []

    for images, targets, tracers in loader:
        images  = images.to(device,  non_blocking=True)
        tracers = tracers.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type,
                                enabled=(device.type == "cuda")):
            preds = model(images, tracers)
        all_preds.append(preds.cpu())
        all_targets.append(targets)

    preds   = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    return (preds - targets).abs().mean().item(), pearson_r(preds, targets)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train mygo_centiloid")
    parser.add_argument("--config", required=True,
                        help="Path to YAML training config "
                             "(see dev/config/train.yaml).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    P, R, M, T, A, L = (cfg["paths"], cfg["run"], cfg["model"],
                        cfg["training"], cfg["augmentation"], cfg["loss"])

    torch.manual_seed(R["seed"])
    np.random.seed(R["seed"])

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Device: {device}  |  AMP: {use_amp}\n")

    # ── Run folder + registry setup ───────────────────────────────────────
    run_dir = make_run_dir(
        run_type="train", model=M["name"],
        base=P["log_dir"], run_name=R["name"],
    )
    checkpoint_dir = P["checkpoint_dir"] or str(run_dir / "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    write_config(run_dir, {"run_type": "train", "config_path": args.config, **cfg})
    print(f"Run dir: {run_dir}")
    print(f"Config : {args.config}\n")

    # ── Datasets ──────────────────────────────────────────────────────────
    train_ds = PETDataset(P["train_csv"], cache=False)
    val_ds   = PETDataset(P["val_csv"], tracer_map=train_ds.tracer_map, cache=False)

    # ── Augmentation setup ────────────────────────────────────────────────
    augment_on = A["enabled"]
    if augment_on and build_train_transform is None:
        print("Augmentation: requested but augmentation.py not found → disabled\n")
        augment_on = False

    if augment_on:
        SMALL_N_TRACERS = {"NAV", "PIB"}
        tid_to_name = {v: k for k, v in train_ds.tracer_map.items()}

        if not A["per_tracer"]:
            train_ds.transform = build_train_transform(strong=False)
            print("Augmentation: uniform standard (per-tracer disabled)\n")
        else:
            train_ds.transform = {
                tid: build_train_transform(
                    strong=(tid_to_name.get(tid, "") in SMALL_N_TRACERS)
                )
                for tid in set(train_ds.tracers.tolist())
            }
            print("Augmentation: per-tracer")
            for tid, name in tid_to_name.items():
                strength = "STRONG" if name in SMALL_N_TRACERS else "standard"
                n = (train_ds.tracers == tid).sum().item()
                print(f"  {name:>4}  n={n:>4}  → {strength}")
            print()
    else:
        print("Augmentation: disabled\n")

    # ── Sampler + DataLoaders ─────────────────────────────────────────────
    sampler = build_weighted_sampler(train_ds.centiloids.numpy())

    train_loader = DataLoader(
        train_ds, batch_size=T["batch_size"], sampler=sampler,
        num_workers=T["num_workers"], pin_memory=use_amp, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=T["batch_size"], shuffle=False,
        num_workers=T["num_workers"], pin_memory=use_amp,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    mean_cl     = float(train_ds.centiloids.mean())
    num_tracers = len(train_ds.tracer_map)

    if M["name"] == "petresnet":
        model = PETResNet(
            num_tracers    = num_tracers,
            emb_dim        = M["emb_dim"],
            dropout_high   = M["dropout_high"],
            dropout_low    = M["dropout_low"],
            mean_centiloid = mean_cl,
        ).to(device)
    elif M["name"] == "petresnet_no_gap":
        model = PETResNetNoGAP(
            num_tracers    = num_tracers,
            emb_dim        = M["emb_dim"],
            dropout_high   = M["dropout_high"],
            dropout_low    = M["dropout_low"],
            mean_centiloid = mean_cl,
        ).to(device)
    elif M["name"] == "petresnet_no_film":
        model = PETResNetNoFiLM(
            num_tracers    = num_tracers,
            emb_dim        = M["emb_dim"],
            dropout_high   = M["dropout_high"],
            dropout_low    = M["dropout_low"],
            mean_centiloid = mean_cl,
        ).to(device)
    elif M["name"] == "petresnet_tracer_norm_only":
        model = PETResNetTracerNormOnly(
            num_tracers    = num_tracers,
            dropout_high   = M["dropout_high"],
            dropout_low    = M["dropout_low"],
            mean_centiloid = mean_cl,
        ).to(device)
    else:
        raise ValueError(f"Unknown model.name: {M['name']!r}")
    print(f"Model: {M['name']}")

    # TracerNorm ablation — force identity and freeze if requested
    if M["freeze_tracer_norm"]:
        with torch.no_grad():
            model.tracer_norm.scale.weight.fill_(1.0)
            model.tracer_norm.shift.weight.fill_(0.0)
        for p in model.tracer_norm.parameters():
            p.requires_grad_(False)
        print("TracerNorm: FROZEN at identity (γ=1, β=0) — ablation mode")
    print()

    model.summary(input_size=(T["batch_size"], 1, 128, 128, 128))

    # ── Loss ──────────────────────────────────────────────────────────────
    loss_kwargs = {}
    if L["name"] in ("centiloid", "huber"):
        loss_kwargs["delta"] = L["delta"]
    if L["name"] == "centiloid":
        loss_kwargs["alpha"] = L["alpha"]
    criterion = get_criterion(L["name"], **loss_kwargs)

    # ── Optimiser + Scheduler ─────────────────────────────────────────────
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=T["lr"], weight_decay=T["weight_decay"])
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    scaler    = torch.amp.GradScaler(enabled=use_amp)

    # ── Training loop ─────────────────────────────────────────────────────
    best_mae     = float("inf")
    patience_cnt = 0
    best_ckpt    = os.path.join(checkpoint_dir, "best_model.pt")
    last_ckpt    = os.path.join(checkpoint_dir, "last_model.pt")

    header = f"{'Epoch':>6} | {'Train Loss':>11} | {'Val MAE':>8} | {'Pearson r':>10} | {'LR':>9}"
    print(header)
    print("─" * len(header))

    for epoch in range(1, T["epochs"] + 1):

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion,
            scaler, device, T["clip_grad"],
        )
        val_mae, val_r = validate(model, val_loader, device)
        scheduler.step(epoch)
        lr = optimizer.param_groups[0]["lr"]

        print(f"{epoch:>6} | {train_loss:>11.4f} | {val_mae:>8.2f} | {val_r:>10.4f} | {lr:>9.2e}")
        append_epoch(run_dir, {
            "epoch": epoch, "train_loss": f"{train_loss:.6f}",
            "val_mae": f"{val_mae:.4f}", "val_r": f"{val_r:.4f}",
            "lr": f"{lr:.3e}",
        })

        ckpt = {
            "epoch":              epoch,
            "model_state_dict":   model.state_dict(),
            "optimizer_state":    optimizer.state_dict(),
            "val_mae":            val_mae,
            "val_r":              val_r,
            "tracer_map":         train_ds.tracer_map,
            "num_tracers":        num_tracers,
            "model":              M["name"],
            "emb_dim":            M["emb_dim"],
            "dropout_high":       M["dropout_high"],
            "dropout_low":        M["dropout_low"],
            "freeze_tracer_norm": M["freeze_tracer_norm"],
            "attention_kernel":   M.get("attention_kernel", 7),
        }
        torch.save(ckpt, last_ckpt)

        if val_mae < best_mae:
            best_mae     = val_mae
            patience_cnt = 0
            torch.save(ckpt, best_ckpt)
            print(f"  ✓ New best MAE: {best_mae:.2f} CL → {best_ckpt}")
        else:
            patience_cnt += 1
            if patience_cnt >= T["patience"]:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no improvement for {T['patience']} epochs)")
                break

    print(f"\nDone. Best Val MAE: {best_mae:.2f} CL  |  checkpoint: {best_ckpt}")

    # ── Final summary to run folder + global registry ─────────────────────
    final_metrics = {
        "run_type":      "train",
        "model":         M["name"],
        "best_val_mae":  best_mae,
        "best_val_r":    val_r,
        "epochs_run":    epoch,
        "early_stopped": patience_cnt >= T["patience"],
        "best_ckpt":     best_ckpt,
        "last_ckpt":     last_ckpt,
    }
    write_metrics(run_dir, final_metrics)
    append_registry({
        "type":    "train",
        "model":   M["name"],
        "run_dir": str(run_dir),
        "ckpt":    best_ckpt,
        "metrics": {"best_val_mae": best_mae, "best_val_r": val_r,
                    "epochs_run": epoch},
        "config":  {"batch_size": T["batch_size"], "lr": T["lr"],
                    "seed": R["seed"], "loss": L["name"],
                    "freeze_tracer_norm": M["freeze_tracer_norm"]},
    })


if __name__ == "__main__":
    main()
