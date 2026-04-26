"""
train_eval.py — Train + Evaluate ACMIL with Patient-Max BCE, Early Stopping,
                Temperature Scaling, and Multi-Seed Ensembling
===============================================================================
Per fold: train K ACMIL models with different random seeds. Each model is
trained with patient-max BCE loss + label smoothing + auxiliary branch-CE +
attention entropy regularization, with early stopping on a held-out val split
(built by create_splits.py). After training, temperature scaling is fit on the
same val split to calibrate the model. The K trained models are then bundled
into a single ensemble checkpoint (one .pth file per fold, containing all K
state_dicts), which predict.py can load and average.

Final output per fold:
  checkpoints/fold_{i}_ensemble.pth   — single file containing K state_dicts
  predictions/fold_{i}_patients.json — per-patient test predictions

After all folds finish, a super-ensemble pooling every seed from every fold
is saved as checkpoints/final_ensemble.pth — this is what you submit to the
leaderboard.

Usage:
  python starter_code/train_eval.py
  python starter_code/train_eval.py --folds 0 --n_seeds 2 --epochs 30
"""

import os
import sys
import json
import copy
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, log_loss
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from model import (
    build_model, build_model_from_config,
    get_patient_dataloader, PatientBagDataset,
    ACMILEnsemble,
    LABEL_MAP, IDX_TO_LABEL, NUM_CLASSES, FEATURE_DIM,
)


# =============================================================================
# SECTION 1: LOSS
# =============================================================================

def patient_max_bce_loss(slide_probs_list, patient_label,
                         label_smoothing=0.0, pos_weight=1.0, eps=1e-7):
    """
    slide_probs_list : list of scalar tensors, each = P(VITUMOR) for one slide
    patient_label    : int (0 or 1)
    pos_weight       : multiplier on the positive-class term (>1 upweights VITUMOR)
    """
    max_p = torch.stack(slide_probs_list).max()
    max_p = max_p.clamp(min=eps, max=1.0 - eps)
    y = float(patient_label)
    # Label smoothing: 0 -> smoothing, 1 -> 1 - smoothing
    y_smooth = y * (1.0 - 2.0 * label_smoothing) + label_smoothing
    loss = -(pos_weight * y_smooth * torch.log(max_p)
             + (1.0 - y_smooth) * torch.log(1.0 - max_p))
    return loss


def attention_neg_entropy(attentions):
    """Sum of negative entropies (smaller = more spread). Minimize to maximize entropy."""
    total = 0.0
    n = 0
    for a in attentions:
        a = a.clamp(min=1e-8)
        total = total + (a * torch.log(a)).sum()
        n += 1
    if n == 0:
        return torch.tensor(0.0)
    return total / n


# =============================================================================
# SECTION 2: TRAIN / EVAL LOOPS
# =============================================================================

def train_one_epoch(model, loader, optimizer, device, args, epoch_desc="train"):
    model.train()
    total_loss = 0.0
    n_patients = 0

    pbar = tqdm(loader, desc=epoch_desc, leave=False,
                dynamic_ncols=True, unit="pt")
    for patient_batch in pbar:
        patient = patient_batch[0]
        optimizer.zero_grad()

        slide_probs_list   = []
        branch_ce_losses   = []
        entropy_losses     = []

        label_t = torch.tensor([patient["patient_label"]],
                               dtype=torch.long, device=device)

        for slide in patient["slides"]:
            features = slide["features"].to(device)
            coords   = slide["coords"].to(device)

            _, aux = model(features, coords)

            # Patient-max BCE uses raw (pre-temperature) softmax probs
            raw_logits = aux["raw_mean_logits"]               # (1, 2)
            p_vi = torch.softmax(raw_logits, dim=-1)[0, 1]    # scalar
            slide_probs_list.append(p_vi)

            # Branch-wise CE aux
            branch_logits = aux["branch_logits"]              # (M, 1, 2)
            for b in range(branch_logits.shape[0]):
                branch_ce_losses.append(
                    F.cross_entropy(branch_logits[b], label_t,
                                    label_smoothing=args.label_smoothing)
                )

            # Attention entropy aux (minimize neg-entropy = maximize entropy)
            entropy_losses.append(attention_neg_entropy(aux["branch_attentions"]))

        loss_main    = patient_max_bce_loss(slide_probs_list,
                                            patient["patient_label"],
                                            label_smoothing=args.label_smoothing,
                                            pos_weight=args._pos_weight)
        loss_branch  = (torch.stack(branch_ce_losses).mean()
                        if branch_ce_losses else torch.tensor(0.0, device=device))
        loss_entropy = (torch.stack(entropy_losses).mean()
                        if entropy_losses else torch.tensor(0.0, device=device))

        loss = (loss_main
                + args.branch_ce_weight * loss_branch
                + args.entropy_weight   * loss_entropy)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_patients += 1
        pbar.set_postfix(loss=f"{total_loss / n_patients:.4f}")

    pbar.close()
    return total_loss / max(n_patients, 1)


@torch.no_grad()
def evaluate_patient_level(model, loader, device, use_temperature=True, clip_eps=0.02,
                           desc="eval"):
    """
    Returns
    -------
    metrics : dict with keys
        "log_loss" (float),
        "auc"      (float or nan),
        "patient_probs"  (list of float — max P(VI) per patient),
        "patient_labels" (list of int),
        "patient_pids"   (list of str),
    """
    model.eval()
    patient_probs = []
    patient_labels = []
    patient_pids = []

    T = (model.temperature.clamp(min=1e-3) if use_temperature
         else torch.tensor(1.0, device=device))

    for patient_batch in tqdm(loader, desc=desc, leave=False,
                              dynamic_ncols=True, unit="pt"):
        patient = patient_batch[0]
        probs_this_patient = []
        for slide in patient["slides"]:
            features = slide["features"].to(device)
            coords   = slide["coords"].to(device)
            _, aux = model(features, coords)
            raw_logits = aux["raw_mean_logits"]
            p = torch.softmax(raw_logits / T, dim=-1)[0, 1].item()
            probs_this_patient.append(p)
        max_p = max(probs_this_patient) if probs_this_patient else 0.5
        # Apply probability clipping to match the inference-time guarantee
        max_p = float(np.clip(max_p, clip_eps, 1.0 - clip_eps))
        patient_probs.append(max_p)
        patient_labels.append(patient["patient_label"])
        patient_pids.append(patient["pid"])

    probs_arr  = np.array(patient_probs)
    labels_arr = np.array(patient_labels)
    probs_2d   = np.stack([1.0 - probs_arr, probs_arr], axis=1)

    ll = log_loss(labels_arr, probs_2d, labels=[0, 1])
    if len(np.unique(labels_arr)) == 2:
        auc = roc_auc_score(labels_arr, probs_arr)
    else:
        auc = float("nan")

    return {
        "log_loss":       ll,
        "auc":            auc,
        "patient_probs":  patient_probs,
        "patient_labels": patient_labels,
        "patient_pids":   patient_pids,
    }


# =============================================================================
# SECTION 3: TEMPERATURE SCALING
# =============================================================================

@torch.no_grad()
def _collect_val_patient_logits(model, loader, device):
    """Cache raw logits per val patient so temperature fitting is cheap."""
    model.eval()
    cache = []
    for patient_batch in tqdm(loader, desc="cache val logits", leave=False,
                              dynamic_ncols=True, unit="pt"):
        patient = patient_batch[0]
        raw_list = []
        for slide in patient["slides"]:
            features = slide["features"].to(device)
            coords   = slide["coords"].to(device)
            _, aux = model(features, coords)
            raw_list.append(aux["raw_mean_logits"].squeeze(0).detach())  # (2,)
        cache.append({
            "raw_logits":    torch.stack(raw_list),         # (n_slides, 2)
            "patient_label": patient["patient_label"],
        })
    return cache


def fit_temperature(model, val_loader, device, max_iter: int = 100, eps: float = 1e-7):
    """
    Fit a single scalar temperature T on the val set to minimize
    per-patient log loss under the max-aggregation rule. The fitted T is
    written back to model.temperature.
    """
    print("  Fitting temperature on val...")
    cache = _collect_val_patient_logits(model, val_loader, device)
    if len(cache) == 0:
        print("    [fit_temperature] empty val set — skipping.")
        return

    T = nn.Parameter(torch.ones(1, device=device))
    optimizer = torch.optim.LBFGS([T], lr=0.1, max_iter=max_iter,
                                  line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        losses = []
        for item in cache:
            raw = item["raw_logits"]                                # (n, 2)
            scaled = raw / T.clamp(min=1e-3)
            p_vi = torch.softmax(scaled, dim=-1)[:, 1]              # (n,)
            max_p = p_vi.max().clamp(min=eps, max=1.0 - eps)
            y = float(item["patient_label"])
            loss = -(y * torch.log(max_p) + (1.0 - y) * torch.log(1.0 - max_p))
            losses.append(loss)
        loss = torch.stack(losses).mean()
        loss.backward()
        return loss

    try:
        optimizer.step(closure)
    except Exception as e:
        print(f"    [fit_temperature] LBFGS failed ({e}); keeping T=1.0")
        return

    fitted = float(T.detach().clamp(min=1e-3).item())
    model.temperature.data = torch.tensor([fitted], device=device)
    print(f"    Fitted temperature T = {fitted:.4f}")


# =============================================================================
# SECTION 4: SINGLE-SEED TRAINING ROUTINE
# =============================================================================

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_pos_weight(args, fold_data):
    """Compute pos_weight from the fold's training set if --pos_weight auto."""
    if isinstance(args.pos_weight, str) and args.pos_weight.lower() == "auto":
        # Count patient-level labels in train split
        pids_seen = {}
        for s in fold_data["train"]:
            pids_seen.setdefault(s["pid"], []).append(s["vi_label"])
        n_pos = sum(1 for lbls in pids_seen.values() if "VITUMOR" in lbls)
        n_neg = sum(1 for lbls in pids_seen.values() if "VITUMOR" not in lbls)
        if n_pos == 0:
            return 1.0
        return float(n_neg) / float(n_pos)
    return float(args.pos_weight)


def train_one_seed(fold_data, seed, args, device):
    print(f"\n  ---- Seed {seed} ----")
    set_seed(seed)

    # Resolve and stash pos_weight for this fold's train set
    args._pos_weight = _resolve_pos_weight(args, fold_data)
    print(f"    pos_weight = {args._pos_weight:.3f}")

    train_loader = get_patient_dataloader(
        fold_data["train"], args.features_dir, shuffle=True,
        num_workers=args.num_workers)
    val_loader = get_patient_dataloader(
        fold_data["val"], args.features_dir, shuffle=False,
        num_workers=max(1, args.num_workers // 2))

    if len(train_loader.dataset) == 0 or len(val_loader.dataset) == 0:
        print("  ERROR: empty train or val dataset.")
        return None

    model = build_model(
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        n_branches=args.n_branches,
        top_k=args.top_k,
        mask_prob=args.mask_prob,
        use_pe=args.use_pe,
        pe_dim=args.pe_dim,
        clip_eps=args.clip_eps,
    ).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    best_val = float("inf")
    best_state = None
    patience = args.patience
    patience_left = patience

    epoch_bar = tqdm(range(1, args.epochs + 1),
                     desc=f"seed {seed}",
                     dynamic_ncols=True, unit="ep")
    for epoch in epoch_bar:
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, args,
            epoch_desc=f"  train ep{epoch:3d}")
        scheduler.step()

        val_metrics = evaluate_patient_level(
            model, val_loader, device,
            use_temperature=False, clip_eps=args.clip_eps,
            desc=f"  val   ep{epoch:3d}")
        val_ll = val_metrics["log_loss"]
        val_auc = val_metrics["auc"]

        improved = val_ll < best_val - 1e-5
        if improved:
            best_val = val_ll
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1

        auc_str = f"{val_auc:.3f}" if not np.isnan(val_auc) else "n/a"
        epoch_bar.set_postfix(
            train=f"{train_loss:.4f}",
            val=f"{val_ll:.4f}",
            best=f"{best_val:.4f}",
            auc=auc_str,
            pat=f"{patience_left}/{patience}",
            star="*" if improved else " ",
        )

        if patience_left <= 0:
            epoch_bar.write(
                f"    Early stopping at epoch {epoch} (best val_ll={best_val:.4f})")
            break
    epoch_bar.close()

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # Reload best and fit temperature on val
    model.load_state_dict(best_state)
    fit_temperature(model, val_loader, device, max_iter=args.temp_max_iter)

    final_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # Report calibrated val metrics
    val_metrics_cal = evaluate_patient_level(
        model, val_loader, device,
        use_temperature=True, clip_eps=args.clip_eps)
    print(f"    post-calibration val_ll={val_metrics_cal['log_loss']:.4f}  "
          f"val_auc={val_metrics_cal['auc']:.4f}")

    return {
        "seed":        seed,
        "state_dict":  final_state,
        "val_log_loss": val_metrics_cal["log_loss"],
        "val_auc":      val_metrics_cal["auc"],
    }


# =============================================================================
# SECTION 5: FOLD RUNNER
# =============================================================================

def build_config_dict(args):
    return {
        "feature_dim": FEATURE_DIM,
        "hidden_dim":  args.hidden_dim,
        "num_classes": NUM_CLASSES,
        "dropout":     args.dropout,
        "n_branches":  args.n_branches,
        "top_k":       args.top_k,
        "mask_prob":   args.mask_prob,
        "use_pe":      args.use_pe,
        "pe_dim":      args.pe_dim,
        "clip_eps":    args.clip_eps,
    }


def build_ensemble_from_states(state_dicts, config, device):
    members = []
    for state in state_dicts:
        m = build_model_from_config(config, verbose=False).to(device)
        m.load_state_dict(state)
        m.eval()
        members.append(m)
    return ACMILEnsemble(members).to(device).eval()


def run_fold(fold_idx, fold_data, args, device):
    print(f"\n{'='*60}")
    print(f"FOLD {fold_idx}  —  "
          f"train={len(fold_data['train'])}  "
          f"val={len(fold_data['val'])}  "
          f"test={len(fold_data['test'])}")
    print(f"{'='*60}")

    seeds = list(range(args.base_seed, args.base_seed + args.n_seeds))
    seed_results = []
    seed_bar = tqdm(seeds, desc=f"fold {fold_idx} seeds",
                    dynamic_ncols=True, unit="seed")
    for seed in seed_bar:
        result = train_one_seed(fold_data, seed, args, device)
        if result is not None:
            seed_results.append(result)
            seed_bar.set_postfix(
                last_val_ll=f"{result['val_log_loss']:.4f}",
                done=f"{len(seed_results)}/{len(seeds)}",
            )
    seed_bar.close()

    if len(seed_results) == 0:
        print("  No seeds completed successfully.")
        return None

    # Build ensemble and evaluate on fold test
    config = build_config_dict(args)
    ensemble = build_ensemble_from_states(
        [r["state_dict"] for r in seed_results], config, device)

    test_loader = get_patient_dataloader(
        fold_data["test"], args.features_dir, shuffle=False,
        num_workers=max(1, args.num_workers // 2))

    class _EnsembleInferWrap(nn.Module):
        """Compatibility wrapper so evaluate_patient_level can use an ensemble."""
        def __init__(self, ens):
            super().__init__()
            self.ens = ens
            self.temperature = torch.ones(1, device=device)
        def forward(self, features, coords=None):
            logits, _ = self.ens(features, coords)
            # Return log-probs as "raw_mean_logits" so evaluate_patient_level's
            # softmax/temperature path still works (T=1 no-op).
            return logits, {"raw_mean_logits": logits}

    wrap = _EnsembleInferWrap(ensemble)
    test_metrics = evaluate_patient_level(
        wrap, test_loader, device,
        use_temperature=False, clip_eps=args.clip_eps)

    print(f"\n  [Fold {fold_idx} ensemble — per-patient test metrics]")
    print(f"    log loss : {test_metrics['log_loss']:.4f}")
    print(f"    AUC      : {test_metrics['auc']:.4f}")

    # Save single-file ensemble checkpoint
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        ckpt_path = os.path.join(args.save_dir, f"fold_{fold_idx}_ensemble.pth")
        torch.save({
            "fold":         fold_idx,
            "seeds":        [r["seed"] for r in seed_results],
            "model_states": [r["state_dict"] for r in seed_results],
            "config":       config,
            "args":         vars(args),
            "metrics": {
                "test_log_loss": test_metrics["log_loss"],
                "test_auc":      test_metrics["auc"],
            },
        }, ckpt_path)
        print(f"  Ensemble checkpoint saved → {ckpt_path}")

    # Save per-patient predictions
    if args.preds_dir:
        os.makedirs(args.preds_dir, exist_ok=True)
        pred_records = []
        for pid, y, p in zip(test_metrics["patient_pids"],
                             test_metrics["patient_labels"],
                             test_metrics["patient_probs"]):
            pred_records.append({
                "pid":          pid,
                "true_label":   IDX_TO_LABEL[int(y)],
                "patient_score": round(float(p), 6),
                "pred_label":   IDX_TO_LABEL[1 if p >= 0.5 else 0],
            })
        preds_path = os.path.join(args.preds_dir, f"fold_{fold_idx}_patients.json")
        with open(preds_path, "w") as f:
            json.dump(pred_records, f, indent=2)
        print(f"  Per-patient predictions saved → {preds_path}")

    return {
        "fold":         fold_idx,
        "log_loss":     test_metrics["log_loss"],
        "auc":          test_metrics["auc"],
        "seed_results": seed_results,
        "ensemble":     ensemble,
        "config":       config,
    }


# =============================================================================
# SECTION 6: MAIN
# =============================================================================

def save_super_ensemble(args, folds_results):
    """Pool every seed from every fold into a single super-ensemble .pth."""
    if len(folds_results) == 0 or not args.save_dir:
        return

    all_states = []
    config = folds_results[0]["config"]
    for fr in folds_results:
        for sr in fr["seed_results"]:
            all_states.append(sr["state_dict"])

    print(f"\n{'='*72}")
    print("FINAL SUPER-ENSEMBLE")
    print(f"{'='*72}")
    print(f"  Pooling {len(all_states)} models "
          f"({len(folds_results)} folds × {len(folds_results[0]['seed_results'])} seeds)")

    final_path = os.path.join(args.save_dir, "final_ensemble.pth")
    torch.save({
        "fold":         -1,
        "seeds":        "super-ensemble-all-folds",
        "model_states": all_states,
        "config":       config,
        "args":         vars(args),
    }, final_path)
    print(f"  Final super-ensemble saved → {final_path}")
    print(f"  (Use this file as CHECKPOINT in predict.sh for leaderboard submission.)")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    splits_dir = Path(args.splits_dir)
    all_fold_files = sorted(splits_dir.glob("fold_*.json"))
    if len(all_fold_files) == 0:
        print(f"ERROR: No fold_*.json files in {splits_dir}. Run create_splits.py first.")
        return

    if args.folds is not None:
        fold_files = [f for f in all_fold_files
                      if int(f.stem.split("_")[1]) in args.folds]
    else:
        fold_files = all_fold_files
    print(f"Running {len(fold_files)} fold(s): {[f.name for f in fold_files]}")

    fold_results = []
    fold_bar = tqdm(fold_files, desc="folds", dynamic_ncols=True, unit="fold")
    for fold_file in fold_bar:
        fold_idx = int(fold_file.stem.split("_")[1])
        with open(fold_file) as f:
            fold_data = json.load(f)
        if "val" not in fold_data or len(fold_data["val"]) == 0:
            fold_bar.write(f"WARNING: fold_{fold_idx}.json has no 'val' split. "
                           f"Re-run create_splits.py with --val_frac > 0.")
            continue
        result = run_fold(fold_idx, fold_data, args, device)
        if result is not None:
            fold_results.append(result)
            fold_bar.set_postfix(
                last_ll=f"{result['log_loss']:.4f}",
                done=f"{len(fold_results)}/{len(fold_files)}",
            )
    fold_bar.close()

    if len(fold_results) == 0:
        print("\nNo folds completed successfully.")
        return

    # CV summary
    print(f"\n{'='*72}")
    print("CROSS-VALIDATION SUMMARY (per-fold ensemble, patient-level)")
    print(f"{'='*72}")
    print(f"  {'Fold':>6}  {'Log Loss':>10}  {'AUC':>8}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*8}")
    for r in fold_results:
        auc = f"{r['auc']:>8.4f}" if not np.isnan(r['auc']) else f"{'n/a':>8}"
        print(f"  {r['fold']:>6d}  {r['log_loss']:>10.4f}  {auc}")
    lls  = [r["log_loss"] for r in fold_results]
    aucs = [r["auc"] for r in fold_results if not np.isnan(r["auc"])]
    print(f"  {'-'*6}  {'-'*10}  {'-'*8}")
    print(f"  {'mean':>6}  {np.mean(lls):>10.4f}  "
          f"{(np.mean(aucs) if aucs else float('nan')):>8.4f}")
    print(f"  {'std':>6}  {np.std(lls):>10.4f}  "
          f"{(np.std(aucs) if aucs else float('nan')):>8.4f}")

    # Save final super-ensemble across all folds × seeds
    save_super_ensemble(args, fold_results)

    print(f"\nDone. Checkpoints saved to: {args.save_dir}")
    if args.preds_dir:
        print(f"      Predictions saved to:  {args.preds_dir}")


# =============================================================================
# SECTION 7: ARGS
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate ACMIL for VI-LUAD",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Paths
    parser.add_argument("--features_dir", type=str,
                        default="/projectnb/medaihack/VI_LUAD_Project/WSI_Data/processed",
                        help="Directory with per-slide .pt feature files.")
    parser.add_argument("--splits_dir",   type=str,
                        default=os.path.join(os.path.dirname(__file__), "splits"))
    parser.add_argument("--save_dir",     type=str,
                        default=os.path.join(os.path.dirname(__file__), "checkpoints"))
    parser.add_argument("--preds_dir",    type=str,
                        default=os.path.join(os.path.dirname(__file__), "predictions"))

    # Training
    parser.add_argument("--epochs",       type=int, default=50)
    parser.add_argument("--patience",     type=int, default=10,
                        help="Early-stopping patience (epochs without val improvement).")
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--pos_weight",   type=str,   default="1.0",
                        help="Multiplier on positive-class (VITUMOR) term in BCE. "
                             "Pass a float (e.g. 1.58) or 'auto' to use n_neg/n_pos "
                             "from each fold's train patients.")
    parser.add_argument("--branch_ce_weight", type=float, default=0.5)
    parser.add_argument("--entropy_weight",   type=float, default=0.01)
    parser.add_argument("--temp_max_iter",    type=int,   default=100)

    # Model
    parser.add_argument("--hidden_dim",   type=int,   default=512)
    parser.add_argument("--dropout",      type=float, default=0.25)
    parser.add_argument("--n_branches",   type=int,   default=5)
    parser.add_argument("--top_k",        type=int,   default=10)
    parser.add_argument("--mask_prob",    type=float, default=0.6)
    parser.add_argument("--use_pe",       type=lambda x: x.lower() != "false", default=True)
    parser.add_argument("--pe_dim",       type=int,   default=64)
    parser.add_argument("--clip_eps",     type=float, default=0.02)

    # Ensembling
    parser.add_argument("--n_seeds",      type=int, default=5,
                        help="Number of random seeds per fold (ensemble size).")
    parser.add_argument("--base_seed",    type=int, default=42)

    # Experiment control
    parser.add_argument("--folds", type=int, nargs="+", default=None,
                        metavar="FOLD_IDX")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader worker processes. Each preloads the "
                             "next patient's .pt file while the GPU trains on "
                             "the current one. Set to 0 for easy debugging.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)