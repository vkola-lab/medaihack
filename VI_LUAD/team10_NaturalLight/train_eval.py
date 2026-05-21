"""
train_eval.py — Training and Evaluation Script for VI-LUAD MIL Model
======================================================================
As a participant, this is the main script you will run. Read this
docstring to understand this training and evaluation pipeline. It:
  1. Loads pre-extracted patch features (from preprocess.py)
  2. Loads the 5-fold cross-validation splits (from preprocess.py)
  3. For each fold: trains the MIL model, evaluates on the test set, saves a checkpoint
  4. Aggregates and prints cross-validation performance

The model (from model.py) is a mean-pooling MIL classifier:
  - Input: a bag of N patch feature vectors (N × feature_dim; default feature_dim=1536)
  - Step 1: Mean pooling → one 1536-dim slide embedding
  - Step 2: MLP head → binary logits (NONVITUMOR=0, VITUMOR=1)

Training details (baseline — minimum setup):
  - Optimizer: Adam with weight decay
  - Loss: cross-entropy (binary)
  - One slide per gradient step (batch_size=1)
  - Fixed number of epochs

Metrics reported (both per-slide and per-patient):
  - Log loss: negative log-likelihood of the true labels under the predicted
              probability distribution. Lower is better; 0 means perfect.
  - AUC: area under the ROC curve. Higher is better; 1.0 is perfect.

  Per-patient aggregation rule:
    A patient is predicted VITUMOR if at least one of their slides is predicted
    VITUMOR (argmax == 1). For AUC, the patient score is max(P(VITUMOR)) across
    all slides — this continuously reflects the "at least one" logic.

Usage:
  # Train on all 5 folds:
  python train_eval.py

  # Quick test on fold 0 only, 5 epochs:
  python train_eval.py --folds 0 --epochs 5

  # Custom paths:
  python train_eval.py \\
      --features_dir ./features \\
      --splits_dir   ./splits \\
      --save_dir     ./checkpoints \\
      --epochs 20 --lr 1e-4

Output:
  ./checkpoints/fold_0.pth           — model checkpoint for fold 0 (best training loss)
  ...
  ./checkpoints/fold_4.pth           — model checkpoint for fold 4
  ./predictions/fold_0.json          — per-slide predictions for fold 0 test set
  ./predictions/fold_0_patients.json — per-patient predictions for fold 0 test set
  ...
  Console output: per-epoch loss + test metrics every --eval_every epochs,
                  cross-validation summary (slide-level and patient-level).

Per-slide prediction JSON format:
  [
    {
      "filename":   "10987.svs",
      "pid":        "109889",
      "true_label": "NONVITUMOR",
      "pred_label": "VITUMOR",
      "probs": {"NONVITUMOR": 0.493, "VITUMOR": 0.507}
    },
    ...
  ]

Per-patient prediction JSON format:
  [
    {
      "pid":          "109889",
      "true_label":   "NONVITUMOR",
      "pred_label":   "VITUMOR",
      "patient_score": 0.507,
      "n_slides":      3
    },
    ...
  ]
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, log_loss

# Import everything we need from model.py.
# sys.path.insert ensures Python finds model.py even if train_eval.py is called
# from a different working directory.
sys.path.insert(0, str(Path(__file__).parent))
from model import (build_model, get_dataloader,
                    LABEL_MAP, IDX_TO_LABEL, NUM_CLASSES, FEATURE_DIM)


# =============================================================================
# SECTION 1: TRAINING
# =============================================================================

def train_one_epoch(model: nn.Module,
                    loader,
                    optimizer: torch.optim.Optimizer,
                    criterion: nn.Module,
                    device: torch.device) -> float:
    """
    Run one full pass over the training set and return the average loss.

    How the MIL training loop works
    --------------------------------
    Each item from the DataLoader is a (features_list, labels) pair where:
      - features_list : Python list of tensors, each shape (N_i, feat_dim).
                        N_i (the number of patches) differs per slide.
      - labels        : torch.LongTensor of shape (batch_size,) — one label
                        per slide in this mini-batch.

    With batch_size=1 (default), features_list always has exactly 1 element.
    We sum the loss across all slides in the mini-batch, average it, then
    call backward() once so that the gradient step covers the whole batch.

    Parameters
    ----------
    model     : MILClassifier in train mode
    loader    : DataLoader built with mil_collate_fn
    optimizer : Adam optimizer
    criterion : CrossEntropyLoss
    device    : CPU or CUDA GPU

    Returns
    -------
    avg_loss : average cross-entropy loss over all slides in this epoch
    """
    model.train()
    total_loss = 0.0
    n_slides = 0

    for features_list, labels in loader:
        optimizer.zero_grad()

        # Accumulate the loss for every slide in this mini-batch.
        # We use a scalar tensor (not a Python float) so that backward()
        # can traverse the full computation graph.
        batch_loss = torch.tensor(0.0, device=device, requires_grad=False)

        for features, label in zip(features_list, labels):
            features = features.to(device)          # (N_i, feat_dim)
            label_t = label.unsqueeze(0).to(device) # (1,) — CrossEntropyLoss expects (B,)

            # Forward pass: get logits (second return value is None for mean pooling).
            logits, _ = model(features)  # logits: (1, 3)

            # Cross-entropy loss: compares logits against the true class index.
            # Internally, it applies log-softmax + NLL, so we don't need to
            # call softmax ourselves.
            loss = criterion(logits, label_t)
            batch_loss = batch_loss + loss

        # Average over slides in the batch before the backward pass.
        # (With batch_size=1 this is a no-op, but it's correct for larger batches.)
        batch_loss = batch_loss / len(features_list)
        batch_loss.backward()
        optimizer.step()

        total_loss += batch_loss.item() * len(features_list)
        n_slides   += len(features_list)

    return total_loss / max(n_slides, 1)


# =============================================================================
# SECTION 2: EVALUATION
# =============================================================================

def evaluate(model: nn.Module,
             loader,
             device: torch.device) -> dict:
    """
    Run inference on every slide in the loader and compute per-slide metrics.

    We collect the softmax probability vector for each slide, then compute:
      - Log loss (cross-entropy): measures how well the predicted probabilities
        match the true labels. Penalises confident wrong predictions heavily.
        sklearn.metrics.log_loss uses base-e logarithm, averaged over slides.
        We pass labels=[0,1] explicitly so that the function does not raise
        an error when only one class appears in a small test fold.
      - AUC (binary): area under the ROC curve for the binary NONVITUMOR/VITUMOR
        task, using P(VITUMOR) as the score. Returns NaN if both classes are not
        present in the test fold.

    Note for participants:
      The primary question is VITUMOR vs. NONVITUMOR. NONTUMOR slides (normal lung
      tissue) have features extracted and can be used creatively — for example, to
      help the model learn what tumor vs. non-tumor tissue looks like. If you use
      NONTUMOR slides in your model, update NUM_CLASSES and LABEL_MAP in model.py
      and replace the binary AUC below with a multi-class version.

    Parameters
    ----------
    model  : MILClassifier in eval mode (set internally)
    loader : DataLoader built with mil_collate_fn (shuffle=False for eval)
    device : CPU or CUDA GPU

    Returns
    -------
    metrics : dict with keys:
        "log_loss"  : float — per-slide cross-entropy on the test-set probabilities
        "auc"       : float — per-slide binary AUC (NaN if only one class present)
        "probs"     : torch.Tensor (N_slides, 2) — softmax probabilities
        "preds"     : torch.Tensor (N_slides,)   — predicted class indices
        "labels"    : torch.Tensor (N_slides,)   — true class indices
        "pids"      : list[str]  — patient ID for each slide (same order as loader)
        "filenames" : list[str]  — slide filename for each slide
    """
    model.eval()

    all_probs  = []
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for features_list, labels in loader:
            for features, label in zip(features_list, labels):
                features = features.to(device)

                # Forward pass — we only need the logits for evaluation
                logits, _ = model(features)  # (1, 2)

                # Convert logits to probabilities with softmax
                probs = torch.softmax(logits, dim=-1).cpu()  # (1, 2)
                pred  = logits.argmax(dim=-1).cpu()          # (1,)

                all_probs.append(probs)
                all_preds.append(pred)
                all_labels.append(label.item())

    # Stack results into matrices for metric computation
    all_probs  = torch.cat(all_probs, dim=0)            # (N_slides, 2)
    all_preds  = torch.cat(all_preds, dim=0)            # (N_slides,)
    all_labels = torch.tensor(all_labels, dtype=torch.long)  # (N_slides,)

    # Patient IDs and filenames — loader.dataset.samples is in the same order
    # as the loader iterates (shuffle=False), so index i in all_probs corresponds
    # to sample i in samples.
    pids      = [s["pid"]      for s in loader.dataset.samples]
    filenames = [s["filename"] for s in loader.dataset.samples]

    # --- Log loss ---
    # We pass labels=[0, 1] so that the function does not raise an error when
    # a test fold contains only one distinct class.
    logloss = log_loss(
        all_labels.numpy(),
        all_probs.numpy(),
        labels=list(range(NUM_CLASSES)),
    )

    # --- Binary AUC ---
    # Use P(VITUMOR) = all_probs[:, 1] as the decision score.
    # roc_auc_score requires both classes to appear in the true labels.
    n_unique = len(torch.unique(all_labels))
    if n_unique == 2:
        auc = roc_auc_score(
            all_labels.numpy(),
            all_probs[:, 1].numpy(),  # P(VITUMOR) as the positive-class score
        )
    else:
        auc = float("nan")
        print(f"    [evaluate] WARNING: only {n_unique}/2 classes "
              f"present in this split — AUC set to NaN.")

    return {
        "log_loss":  logloss,
        "auc":       auc,
        "probs":     all_probs,
        "preds":     all_preds,
        "labels":    all_labels,
        "pids":      pids,
        "filenames": filenames,
    }


def aggregate_patient_predictions(pids: list,
                                   probs: torch.Tensor,
                                   true_labels: torch.Tensor) -> dict:
    """
    Aggregate per-slide predictions to the patient level.

    Why patient-level evaluation?
    ------------------------------
    The clinical question is whether a patient has vascular invasion, not
    whether a single slide does. A patient may have multiple slides; the
    baseline aggregation rule is:

      A patient is predicted VITUMOR if at least one of their slides is
      predicted VITUMOR (argmax == 1).

    For continuous metrics like AUC, we need a single score per patient
    rather than a binary decision. We use max(P(VITUMOR)) across all slides
    as the patient score — a high max value means the model found at least
    one slide strongly suggesting vascular invasion, which mirrors the
    "at least one" binary rule.

    Threshold note:
      The argmax threshold of 0.5 (i.e., pred = 1 if P(VITUMOR) > 0.5) is
      the natural default for cross-entropy–trained models. If your model is
      miscalibrated or the classes are imbalanced, you may find a different
      threshold gives better patient-level accuracy — this is worth exploring.

    Parameters
    ----------
    pids        : list of patient IDs (one per slide, same order as probs/labels)
    probs       : torch.Tensor (N_slides, 2) — per-slide softmax probabilities
    true_labels : torch.Tensor (N_slides,)   — per-slide true class indices

    Returns
    -------
    dict with keys:
        "patient_auc"      : float — binary AUC using max P(VITUMOR) per patient
        "patient_accuracy" : float — fraction of patients correctly classified
        "patient_log_loss" : float — log loss using max P(VITUMOR) per patient
        "patient_pids"     : list[str] — one entry per patient (sorted)
        "patient_labels"   : list[int] — true label per patient (0 or 1)
        "patient_scores"   : list[float] — max P(VITUMOR) per patient
        "patient_preds"    : list[int]   — predicted label per patient (0 or 1)
    """
    from collections import defaultdict

    # Group slide probs and labels by patient ID.
    # All slides from the same patient share the same true label.
    pid_to_probs  = defaultdict(list)
    pid_to_label  = {}
    for pid, prob, label in zip(pids, probs, true_labels.tolist()):
        pid_to_probs[pid].append(prob)
        pid_to_label[pid] = int(label)

    patient_pids   = sorted(pid_to_probs.keys())
    patient_scores = []   # max P(VITUMOR) — continuous score for AUC
    patient_preds  = []   # 1 if max P(VITUMOR) >= 0.5, else 0
    patient_labels = []

    for pid in patient_pids:
        slide_probs   = torch.stack(pid_to_probs[pid])  # (n_slides, 2)
        vitumor_probs = slide_probs[:, 1]               # P(VITUMOR) per slide
        max_score     = vitumor_probs.max().item()
        pred          = 1 if max_score >= 0.5 else 0
        patient_scores.append(max_score)
        patient_preds.append(pred)
        patient_labels.append(pid_to_label[pid])

    labels_arr = np.array(patient_labels)
    scores_arr = np.array(patient_scores)
    preds_arr  = np.array(patient_preds)

    # Binary AUC using the continuous max-score
    if len(np.unique(labels_arr)) == 2:
        patient_auc = roc_auc_score(labels_arr, scores_arr)
    else:
        patient_auc = float("nan")
        print("    [aggregate_patient_predictions] WARNING: only one patient "
              "class present in this fold — patient AUC set to NaN.")

    patient_acc = float((labels_arr == preds_arr).mean())

    # Log loss: construct a 2-column probability matrix from the scalar scores
    patient_ll = log_loss(
        labels_arr,
        np.stack([1 - scores_arr, scores_arr], axis=1),
        labels=[0, 1],
    )

    return {
        "patient_auc":      patient_auc,
        "patient_accuracy": patient_acc,
        "patient_log_loss": patient_ll,
        "patient_pids":     patient_pids,
        "patient_labels":   patient_labels,
        "patient_scores":   patient_scores,
        "patient_preds":    patient_preds,
    }


# =============================================================================
# SECTION 3: SINGLE-FOLD RUNNER
# =============================================================================

def run_fold(fold_idx: int,
             fold_data: dict,
             args: argparse.Namespace,
             device: torch.device):
    """
    Train and evaluate the model on a single cross-validation fold.

    Steps:
      1. Build DataLoaders for train and test sets from this fold's records.
      2. Instantiate a fresh model and Adam optimizer.
      3. Train for args.epochs epochs; print test metrics every args.eval_every epochs.
      4. Load the checkpoint with the best (lowest) training loss.
      5. Run final evaluation: print log loss and AUC.
      6. Save the best checkpoint to args.save_dir/fold_{fold_idx}.pth.

    Parameters
    ----------
    fold_idx  : fold number (0–4)
    fold_data : dict with "train" and "test" keys, each a list of slide records
    args      : parsed command-line arguments
    device    : CPU or CUDA GPU

    Returns
    -------
    dict with fold summary metrics, or None if the fold's dataset is empty.
    """
    train_records = fold_data["train"]
    test_records  = fold_data["test"]

    print(f"\n{'='*60}")
    print(f"FOLD {fold_idx}  —  "
          f"{len(train_records)} train slides / {len(test_records)} test slides")
    print(f"{'='*60}")

    # --- Build DataLoaders ---
    print("\n[Data loading]")
    train_loader = get_dataloader(
        train_records,
        features_dir=args.features_dir,
        batch_size=args.batch_size,
        shuffle=True,    # shuffle training order every epoch
    )
    test_loader = get_dataloader(
        test_records,
        features_dir=args.features_dir,
        batch_size=1,    # one slide at a time for evaluation
        shuffle=False,   # keep order for reproducibility
    )

    # Abort gracefully if features haven't been extracted yet
    if len(train_loader.dataset) == 0 or len(test_loader.dataset) == 0:
        print("  ERROR: train or test dataset is empty. "
              "Did preprocess.py finish extracting features?")
        return None

    # --- Model, optimizer, loss ---
    # A fresh model is created for every fold so they don't share weights.
    model = build_model(hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)

    # Adam optimizer: a good default for MIL models.
    #   lr          — step size. 1e-4 works well for most MIL setups.
    #   weight_decay — L2 penalty on weights; helps prevent overfitting on
    #                  small datasets like ours (~100 training slides).
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Cross-entropy loss: standard choice for multi-class classification.
    # If you observe one class dominating predictions (e.g., the model always
    # predicts VITUMOR), consider adding class weights here:
    #   counts = [n_VITUMOR, n_NONVITUMOR, n_NONTUMOR]  (training slides)
    #   weights = 1 / torch.tensor(counts, dtype=torch.float)
    #   criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    criterion = nn.CrossEntropyLoss()

    # --- Training loop ---
    print(f"\n[Training]  epochs={args.epochs}  lr={args.lr}  "
          f"weight_decay={args.weight_decay}  batch_size={args.batch_size}")
    print(f"  (printing test metrics every {args.eval_every} epochs)")

    best_train_loss   = float("inf")
    best_model_state  = None  # will hold a CPU copy of the best weights

    for epoch in range(1, args.epochs + 1):
        # --- One training pass ---
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # --- Periodic evaluation ---
        # Evaluating every epoch adds overhead (each test pass processes all
        # test slides). We only do it every eval_every epochs and at the end.
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            metrics = evaluate(model, test_loader, device)
            auc_str = f"{metrics['auc']:.4f}" if not np.isnan(metrics['auc']) else " n/a "
            print(f"  Epoch {epoch:3d}/{args.epochs}  "
                  f"train_loss={train_loss:.4f}  "
                  f"test_logloss={metrics['log_loss']:.4f}  "
                  f"test_auc={auc_str}")
        else:
            print(f"  Epoch {epoch:3d}/{args.epochs}  train_loss={train_loss:.4f}")

        # Track the best model by training loss.
        # We save a CPU copy of the weights — this is memory-efficient and
        # lets us reload the best state at the end regardless of device.
        if train_loss < best_train_loss:
            best_train_loss  = train_loss
            best_model_state = {k: v.detach().cpu().clone()
                                for k, v in model.state_dict().items()}

    # --- Final evaluation with best weights ---
    print(f"\n[Evaluation]  Loading best checkpoint (train_loss={best_train_loss:.4f})")
    model.load_state_dict(best_model_state)
    final = evaluate(model, test_loader, device)

    # Patient-level aggregation
    patient = aggregate_patient_predictions(
        final["pids"], final["probs"], final["labels"])

    # Per-slide summary
    auc_str = f"{final['auc']:.4f}" if not np.isnan(final['auc']) else "n/a"
    print(f"\n  [Per-slide]")
    print(f"    Log loss : {final['log_loss']:.4f}")
    print(f"    AUC      : {auc_str}")

    # Per-patient summary
    p_auc_str = f"{patient['patient_auc']:.4f}" if not np.isnan(patient['patient_auc']) else "n/a"
    print(f"\n  [Per-patient]")
    print(f"    Log loss : {patient['patient_log_loss']:.4f}")
    print(f"    AUC      : {p_auc_str}")
    print(f"    Accuracy : {patient['patient_accuracy']:.4f}")

    # --- Save checkpoint and predictions ---
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

        # Save model checkpoint (includes both slide-level and patient-level metrics).
        ckpt_path = os.path.join(args.save_dir, f"fold_{fold_idx}.pth")
        torch.save({
            "fold":             fold_idx,
            "model_state_dict": best_model_state,
            "args":             vars(args),
            "metrics": {
                "slide_log_loss":   final["log_loss"],
                "slide_auc":        final["auc"],
                "patient_log_loss": patient["patient_log_loss"],
                "patient_auc":      patient["patient_auc"],
                "patient_accuracy": patient["patient_accuracy"],
            },
        }, ckpt_path)
        print(f"\n  Checkpoint saved → {ckpt_path}")
        print(f"  (To reload: torch.load('{ckpt_path}')['model_state_dict'])")

    # Save per-slide and per-patient predictions as JSON files.
    if args.preds_dir:
        os.makedirs(args.preds_dir, exist_ok=True)

        # --- Per-slide predictions ---
        # test_loader.dataset.samples lists the slides in the same order as the
        # loader iterates them (shuffle=False), so index i in final['preds']
        # corresponds to sample i in the dataset.
        preds_path = os.path.join(args.preds_dir, f"fold_{fold_idx}.json")
        probs_np = final["probs"].numpy()   # (N_slides, 2)
        preds_np = final["preds"].numpy()   # (N_slides,)
        samples  = test_loader.dataset.samples

        pred_records = []
        for i, sample in enumerate(samples):
            pred_records.append({
                "filename":   sample["filename"],
                "pid":        sample["pid"],
                "true_label": sample["vi_label"],
                "pred_label": IDX_TO_LABEL[int(preds_np[i])],
                "probs": {
                    IDX_TO_LABEL[c]: round(float(probs_np[i, c]), 4)
                    for c in range(NUM_CLASSES)
                },
            })

        with open(preds_path, "w") as fout:
            json.dump(pred_records, fout, indent=2)
        print(f"  Per-slide predictions  saved → {preds_path}")

        # --- Per-patient predictions ---
        patient_preds_path = os.path.join(args.preds_dir, f"fold_{fold_idx}_patients.json")
        patient_records = []
        for pid, true_lbl, score, pred in zip(
            patient["patient_pids"],
            patient["patient_labels"],
            patient["patient_scores"],
            patient["patient_preds"],
        ):
            patient_records.append({
                "pid":           pid,
                "true_label":    IDX_TO_LABEL[true_lbl],
                "pred_label":    IDX_TO_LABEL[pred],
                "patient_score": round(score, 4),  # max P(VITUMOR) across slides
                "n_slides":      len([s for s in samples if s["pid"] == pid]),
            })

        with open(patient_preds_path, "w") as fout:
            json.dump(patient_records, fout, indent=2)
        print(f"  Per-patient predictions saved → {patient_preds_path}")

    return {
        "fold":             fold_idx,
        "log_loss":         final["log_loss"],
        "auc":              final["auc"],
        "patient_log_loss": patient["patient_log_loss"],
        "patient_auc":      patient["patient_auc"],
        "patient_accuracy": patient["patient_accuracy"],
    }


# =============================================================================
# SECTION 4: MAIN — LOOP OVER FOLDS AND AGGREGATE
# =============================================================================

def main(args: argparse.Namespace):
    """
    Load split files, run training+evaluation on each requested fold,
    then print a cross-validation summary.
    """
    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # --- Locate split files ---
    splits_dir = Path(args.splits_dir)
    all_fold_files = sorted(splits_dir.glob("fold_*.json"))

    if len(all_fold_files) == 0:
        print(f"ERROR: No fold_*.json files found in '{splits_dir}'. "
              f"Run preprocess.py first (or check --splits_dir).")
        return

    # Optionally restrict to specific folds (e.g. --folds 0 2 4)
    if args.folds is not None:
        fold_files = [f for f in all_fold_files
                      if int(f.stem.split("_")[1]) in args.folds]
    else:
        fold_files = all_fold_files

    print(f"Running {len(fold_files)} fold(s): {[f.name for f in fold_files]}")

    # --- Run each fold ---
    fold_results = []
    for fold_file in fold_files:
        fold_idx = int(fold_file.stem.split("_")[1])
        with open(fold_file) as f:
            fold_data = json.load(f)
        result = run_fold(fold_idx, fold_data, args, device)
        if result is not None:
            fold_results.append(result)

    # --- Cross-validation summary ---
    if len(fold_results) == 0:
        print("\nNo folds completed successfully.")
        return

    print(f"\n{'='*72}")
    print("CROSS-VALIDATION SUMMARY")
    print(f"{'='*72}")

    # Per-slide metrics
    print(f"\n  [Per-slide]")
    print(f"  {'Fold':>6}  {'Log Loss':>10}  {'AUC':>8}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*8}")
    for r in fold_results:
        auc_str = f"{r['auc']:>8.4f}" if not np.isnan(r['auc']) else f"{'n/a':>8}"
        print(f"  {r['fold']:>6d}  {r['log_loss']:>10.4f}  {auc_str}")
    if len(fold_results) > 1:
        lls  = [r["log_loss"] for r in fold_results]
        aucs = [r["auc"] for r in fold_results if not np.isnan(r["auc"])]
        print(f"  {'-'*6}  {'-'*10}  {'-'*8}")
        print(f"  {'mean':>6}  {np.mean(lls):>10.4f}  "
              f"{np.mean(aucs) if aucs else float('nan'):>8.4f}")
        print(f"  {'std':>6}  {np.std(lls):>10.4f}  "
              f"{np.std(aucs) if aucs else float('nan'):>8.4f}")

    # Per-patient metrics
    print(f"\n  [Per-patient]")
    print(f"  {'Fold':>6}  {'Log Loss':>10}  {'AUC':>8}  {'Accuracy':>10}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*8}  {'-'*10}")
    for r in fold_results:
        p_auc_str = f"{r['patient_auc']:>8.4f}" if not np.isnan(r['patient_auc']) else f"{'n/a':>8}"
        print(f"  {r['fold']:>6d}  {r['patient_log_loss']:>10.4f}  "
              f"{p_auc_str}  {r['patient_accuracy']:>10.4f}")
    if len(fold_results) > 1:
        p_lls  = [r["patient_log_loss"] for r in fold_results]
        p_aucs = [r["patient_auc"] for r in fold_results if not np.isnan(r["patient_auc"])]
        p_accs = [r["patient_accuracy"] for r in fold_results]
        print(f"  {'-'*6}  {'-'*10}  {'-'*8}  {'-'*10}")
        print(f"  {'mean':>6}  {np.mean(p_lls):>10.4f}  "
              f"{np.mean(p_aucs) if p_aucs else float('nan'):>8.4f}  "
              f"{np.mean(p_accs):>10.4f}")
        print(f"  {'std':>6}  {np.std(p_lls):>10.4f}  "
              f"{np.std(p_aucs) if p_aucs else float('nan'):>8.4f}  "
              f"{np.std(p_accs):>10.4f}")

    print(f"\nDone. Checkpoints saved to: {args.save_dir}")
    if args.preds_dir:
        print(f"      Predictions saved to:  {args.preds_dir}")


# =============================================================================
# SECTION 5: ARGUMENT PARSING AND ENTRY POINT
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the mean-pooling MIL model for VI-LUAD",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Paths ---
    path_group = parser.add_argument_group("Paths")
    path_group.add_argument(
        "--features_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "features"),
        help="Directory containing <slide>.pt feature files (from preprocess.py)",
    )
    path_group.add_argument(
        "--splits_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "splits"),
        help="Directory containing fold_0.json … fold_4.json (from preprocess.py)",
    )
    path_group.add_argument(
        "--save_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "checkpoints"),
        help="Directory to save model checkpoints (one .pth file per fold)",
    )
    path_group.add_argument(
        "--preds_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "predictions"),
        help="Directory to save per-slide prediction JSON files (one per fold). "
             "Set to empty string to skip saving predictions.",
    )

    # --- Training hyperparameters ---
    train_group = parser.add_argument_group("Training")
    train_group.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs per fold",
    )
    train_group.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Adam learning rate",
    )
    train_group.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Adam weight decay (L2 regularization coefficient)",
    )
    train_group.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Slides per gradient step. 1 is recommended for MIL with variable bag sizes.",
    )

    # --- Model hyperparameters ---
    model_group = parser.add_argument_group("Model")
    model_group.add_argument(
        "--hidden_dim",
        type=int,
        default=256,
        help="Hidden layer size for the MLP head",
    )
    model_group.add_argument(
        "--dropout",
        type=float,
        default=0.25,
        help="Dropout probability in the MLP classification head",
    )

    # --- Experiment control ---
    misc_group = parser.add_argument_group("Experiment control")
    misc_group.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=None,
        metavar="FOLD_IDX",
        help="Fold indices to run (e.g. --folds 0 1). Default: run all 5 folds.",
    )
    misc_group.add_argument(
        "--eval_every",
        type=int,
        default=5,
        help="Print test metrics every this many epochs (also always printed at final epoch)",
    )
    misc_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (affects model initialization and data shuffling)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Set random seeds for reproducibility.
    # Note: the seed here mainly affects model weight initialization
    # and the order in which training slides are shuffled each epoch.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    main(args)