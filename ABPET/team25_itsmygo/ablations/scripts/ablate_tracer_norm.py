"""
ablate_tracer_norm.py  —  EXPERIMENTAL, not part of the main pipeline

Runs inference twice on the same checkpoint:
  (1) baseline — TracerNorm loaded from checkpoint
  (2) ablated  — TracerNorm forced to identity (γ=1, β=0)

Reports MAE / RMSE / Pearson r overall and per tracer, plus Δ between the two
modes, so you can judge whether TracerNorm is actually doing anything.

Requires the val CSV to have CENTILOIDS (ground truth).

Usage:
    python ablations/scripts/ablate_tracer_norm.py \\
        --csv        /projectnb/medaihack/ABPET/data/val.csv \\
        --checkpoint checkpoints/tracer_norm_only/best_model.pt
"""

import argparse
import os
import sys
import numpy as np
import torch
from scipy.stats import pearsonr
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mygo_centiloid import (
    PETDataset,
    PETResNet, PETResNetNoGAP, PETResNetNoFiLM, PETResNetTracerNormOnly,
)


@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()
    preds, truths, tracers = [], [], []
    for images, targets, tids in loader:
        images = images.to(device, non_blocking=True)
        tids_d = tids.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type,
                                enabled=(device.type == "cuda")):
            y = model(images, tids_d)
        preds.append(y.float().cpu())
        truths.append(targets)
        tracers.append(tids)
    return (torch.cat(preds).numpy(),
            torch.cat(truths).numpy(),
            torch.cat(tracers).numpy())


def metrics(y_true, y_pred):
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    r, _ = pearsonr(y_true, y_pred) if len(y_true) > 1 else (float("nan"), None)
    return {"n": len(y_true), "MAE": mae, "RMSE": rmse, "r": float(r)}


def build_model(ckpt, device):
    name = ckpt.get("model", "petresnet")
    kw = dict(
        num_tracers=ckpt["num_tracers"],
        dropout_high=ckpt.get("dropout_high", 0.4),
        dropout_low=ckpt.get("dropout_low", 0.2),
    )
    if name == "petresnet_tracer_norm_only":
        model = PETResNetTracerNormOnly(**kw).to(device)
    elif name == "petresnet_no_gap":
        model = PETResNetNoGAP(emb_dim=ckpt.get("emb_dim", 32), **kw).to(device)
    elif name == "petresnet_no_film":
        model = PETResNetNoFiLM(emb_dim=ckpt.get("emb_dim", 32), **kw).to(device)
    else:
        model = PETResNet(emb_dim=ckpt.get("emb_dim", 32), **kw).to(device)

    state = {k.replace("_orig_mod.", ""): v
             for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(state)
    return model, name


def neutralize_tracer_norm(model):
    with torch.no_grad():
        model.tracer_norm.scale.weight.fill_(1.0)
        model.tracer_norm.shift.weight.fill_(0.0)


def print_row(label, m):
    print(f"  {label:<18} {m['n']:>5} "
          f"{m['MAE']:>9.3f} {m['RMSE']:>9.3f} {m['r']:>8.3f}")


def print_delta_row(label, base, abl):
    d_mae = abl["MAE"] - base["MAE"]
    d_rmse = abl["RMSE"] - base["RMSE"]
    d_r = abl["r"] - base["r"]
    arrow = "↑" if d_mae > 0 else ("↓" if d_mae < 0 else "=")
    print(f"  {label:<18} {base['n']:>5} "
          f"{d_mae:>+9.3f} {d_rmse:>+9.3f} {d_r:>+8.3f}  {arrow}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--csv", required=True, help="val.csv (needs CENTILOIDS)")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    dataset = PETDataset(args.csv, tracer_map=ckpt["tracer_map"])
    if not dataset.has_targets:
        print("[!] CSV has no CENTILOIDS column — ablation needs ground truth.")
        sys.exit(1)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    model, model_name = build_model(ckpt, device)
    print(f"Model: {model_name}")
    print(f"Checkpoint: {args.checkpoint}\n")

    # Snapshot so we can restore after ablation (defensive — we don't reuse,
    # but it keeps intent explicit).
    orig_scale = model.tracer_norm.scale.weight.detach().clone()
    orig_shift = model.tracer_norm.shift.weight.detach().clone()
    print("TracerNorm learned params:")
    print(f"  scale (γ): {orig_scale.flatten().tolist()}")
    print(f"  shift (β): {orig_shift.flatten().tolist()}\n")

    # ── Baseline ──
    print("Running BASELINE inference (TracerNorm active)...")
    y_pred_b, y_true, tids = run_inference(model, loader, device)

    # ── Ablated ──
    print("Running ABLATED inference (TracerNorm → identity)...")
    neutralize_tracer_norm(model)
    y_pred_a, _, _ = run_inference(model, loader, device)

    # Restore (in case you import this and reuse the model)
    with torch.no_grad():
        model.tracer_norm.scale.weight.copy_(orig_scale)
        model.tracer_norm.shift.weight.copy_(orig_shift)

    # ── Per-tracer breakdown ──
    inv_map = {v: k for k, v in dataset.tracer_map.items()}
    tracers_sorted = [inv_map[i] for i in sorted(inv_map.keys())]

    header = f"  {'Subset':<18} {'N':>5} {'MAE':>9} {'RMSE':>9} {'r':>8}"
    bar = "  " + "-" * 52

    print("\n" + "=" * 56)
    print("  BASELINE  (TracerNorm active)")
    print("=" * 56)
    print(header); print(bar)
    base_all = metrics(y_true, y_pred_b)
    print_row("ALL", base_all)
    base_per = {}
    for t in tracers_sorted:
        idx = dataset.tracer_map[t]
        m = tids == idx
        if m.sum() < 2: continue
        base_per[t] = metrics(y_true[m], y_pred_b[m])
        print_row(t, base_per[t])

    print("\n" + "=" * 56)
    print("  ABLATED   (γ=1, β=0)")
    print("=" * 56)
    print(header); print(bar)
    abl_all = metrics(y_true, y_pred_a)
    print_row("ALL", abl_all)
    abl_per = {}
    for t in tracers_sorted:
        idx = dataset.tracer_map[t]
        m = tids == idx
        if m.sum() < 2: continue
        abl_per[t] = metrics(y_true[m], y_pred_a[m])
        print_row(t, abl_per[t])

    print("\n" + "=" * 56)
    print("  Δ  (ablated − baseline) — positive MAE/RMSE means ablation HURT")
    print("=" * 56)
    print(f"  {'Subset':<18} {'N':>5} {'ΔMAE':>9} {'ΔRMSE':>9} {'Δr':>8}")
    print(bar)
    print_delta_row("ALL", base_all, abl_all)
    for t in tracers_sorted:
        if t in base_per and t in abl_per:
            print_delta_row(t, base_per[t], abl_per[t])

    # ── Verdict line ──
    d = abl_all["MAE"] - base_all["MAE"]
    print("\nVerdict on overall ΔMAE:")
    if d < 1.0:
        print(f"  ΔMAE = {d:+.2f} CL  →  TracerNorm contribution negligible.")
    elif d < 5.0:
        print(f"  ΔMAE = {d:+.2f} CL  →  TracerNorm contributes modestly.")
    else:
        print(f"  ΔMAE = {d:+.2f} CL  →  TracerNorm is load-bearing.")


if __name__ == "__main__":
    main()
