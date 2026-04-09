"""
predict.py - Leaderboard Prediction for VI-LUAD
================================================
Runs inference on a held-out subset of the external test set (from 2 different
institutions) and reports per-patient log loss (the leaderboard metric).

This script is called by predict.sh, which the organizers will run on your
behalf. You do NOT need to run this script yourself (you will not have access
to the test data). See predict.sh for what to fill in.

IMPORTANT - If you changed the model architecture:
  You MUST update SECTION 1 (load_checkpoint) to load your model correctly.
  Make sure your model takes a feature tensor of shape (N, 1536) as input and
  returns logits of shape (1, 2) as the first element of the output tuple.
  The rest of the script (inference, aggregation, metrics) should work as-is.

Patient-level aggregation
-------------------------
Each slide produces P(VITUMOR) from the model's softmax output.
A patient's score is max(P(VITUMOR)) across all their slides.
The leaderboard metric is per-patient log loss.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import log_loss

sys.path.insert(0, str(Path(__file__).parent))
from model import build_model


# ============================================================================
# SECTION 1: LOAD MODEL CHECKPOINT
# ============================================================================
# If you used a different model architecture, update this function to
# import and instantiate your model instead of the baseline MILClassifier.

def load_checkpoint(checkpoint_path: str, device: torch.device,
                    hidden_dim: int, dropout: float):
    """
    Load a single checkpoint and return the model in eval mode.

    If you changed the model architecture, update this function:
      1. Import your model class instead of (or in addition to) build_model.
      2. Instantiate your model with the correct architecture/hyperparameters.
      3. Load the checkpoint weights into your model.
    """
    model = build_model(hidden_dim=hidden_dim, dropout=dropout).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  Loaded: {checkpoint_path}")
    return model


# ============================================================================
# SECTION 2: INFERENCE (do not modify)
# ============================================================================

def run_inference(model, test_metadata, features_dir: str,
                  device: torch.device):
    """
    Run inference on all leaderboard slides.

    For each slide, we load the pre-extracted features and pass them through
    the model to get a prediction.

    Returns a list of dicts, one per slide, with keys:
      name, institution, pid, vi_label, feature_file, prob_vitumor
    """
    features_dir = Path(features_dir)
    results = []
    skipped = 0

    for record in test_metadata:
        feat_path = features_dir / record["feature_file"]
        if not feat_path.exists():
            print(f"  WARNING: missing features for {record['name']} - skipping")
            skipped += 1
            continue

        # Load features (each .pt file is a dict with "features" and "coords")
        try:
            data = torch.load(feat_path, weights_only=True)
        except TypeError:
            data = torch.load(feat_path)
        features = data["features"].to(device)

        # Forward pass
        with torch.no_grad():
            logits, _ = model(features)
            probs = F.softmax(logits, dim=-1).squeeze(0)  # (2,)

        results.append({
            "name":            record["name"],
            "institution":     record["institution"],
            "pid":             record["pid"],
            "vi_label":        record["vi_label"],
            "feature_file":    record["feature_file"],
            "prob_vitumor":    round(probs[1].item(), 6),
        })

    if skipped:
        print(f"  WARNING: {skipped} slides skipped (missing feature files).")
    print(f"  Inference done: {len(results)} slides processed.")
    return results


# ============================================================================
# SECTION 3: PATIENT-LEVEL AGGREGATION (do not modify)
# ============================================================================

def aggregate_patients(slide_results):
    """
    Aggregate per-slide predictions to per-patient predictions.

    A patient may have multiple slides. The aggregation rule is:
      P(VITUMOR) for a patient = max(P(VITUMOR)) across their slides.

    This "at least one" logic reflects the clinical interpretation: if any
    slide shows evidence of vascular invasion, the patient is VI-positive.
    """
    by_patient = defaultdict(list)
    for r in slide_results:
        by_patient[(r["institution"], r["pid"])].append(r)

    patient_results = []
    for (institution, pid), slides in sorted(by_patient.items()):
        max_prob_vi = max(s["prob_vitumor"] for s in slides)
        # True label: if any slide is VITUMOR the patient is VITUMOR
        true_labels = set(s["vi_label"] for s in slides)
        true_label = "VITUMOR" if "VITUMOR" in true_labels else "NONVITUMOR"

        patient_results.append({
            "institution":  institution,
            "pid":          pid,
            "vi_label":     true_label,
            "n_slides":     len(slides),
            "prob_vitumor": round(max_prob_vi, 6),
        })

    return patient_results


# ============================================================================
# SECTION 4: METRICS (do not modify)
# ============================================================================

def compute_log_loss(results, prob_key="prob_vitumor"):
    """Compute log loss from a list of prediction dicts."""
    y_true = [1 if r["vi_label"] == "VITUMOR" else 0 for r in results]
    y_prob = [r[prob_key] for r in results]
    # Clip probabilities to avoid log(0)
    y_prob_clipped = [max(1e-7, min(1 - 1e-7, p)) for p in y_prob]
    probs_2d = [[1 - p, p] for p in y_prob_clipped]
    return log_loss(y_true, probs_2d, labels=[0, 1])


# ============================================================================
# SECTION 5: MAIN (do not modify)
# ============================================================================

LEADERBOARD_OUTPUT_ROOT = "/projectnb/medaihack/VI_LUAD_Project/private/team_leaderboard_outputs"


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Resolve output directory from team name
    preds_dir = os.path.join(LEADERBOARD_OUTPUT_ROOT, args.team)

    # Load leaderboard metadata
    print(f"\n[Loading leaderboard metadata]")
    with open(args.test_metadata) as f:
        test_metadata = json.load(f)
    print(f"  {len(test_metadata)} slides in leaderboard set")

    # Load checkpoint
    print(f"\n[Loading checkpoint]")
    model = load_checkpoint(args.checkpoint, device,
                            args.hidden_dim, args.dropout)

    # Run inference
    print(f"\n[Running inference on leaderboard slides]")
    slide_results = run_inference(model, test_metadata, args.features_dir,
                                 device)

    # Patient aggregation
    print(f"\n[Aggregating per-patient predictions]")
    patient_results = aggregate_patients(slide_results)
    print(f"  {len(patient_results)} unique patients")

    # Log loss (per-patient log loss is the leaderboard metric)
    patient_ll = compute_log_loss(patient_results)

    print(f"\n[Leaderboard Log Loss]")
    print(f"  Per-patient: {patient_ll:.4f}")

    # Save predictions and metrics
    os.makedirs(preds_dir, exist_ok=True)
    slides_path = Path(preds_dir) / "leaderboard_slides.json"
    patients_path = Path(preds_dir) / "leaderboard_patients.json"
    metrics_path = Path(preds_dir) / "leaderboard_metrics.json"

    with open(slides_path, "w") as f:
        json.dump(slide_results, f, indent=2)
    with open(patients_path, "w") as f:
        json.dump(patient_results, f, indent=2)
    with open(metrics_path, "w") as f:
        json.dump({
            "patient_log_loss": round(patient_ll, 4),
        }, f, indent=2)

    print(f"\n  Saved: {slides_path}")
    print(f"  Saved: {patients_path}")
    print(f"  Saved: {metrics_path}")


# ============================================================================
# SECTION 6: ARGUMENT PARSING (do not modify)
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference on leaderboard test slides and save predictions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--team",
        required=True,
        help="Your team's directory name (same as YOUR_TEAM in README).",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to your model checkpoint file (e.g. checkpoints/fold_0.pth)",
    )
    parser.add_argument(
        "--test_metadata",
        default="/projectnb/medaihack/VI_LUAD_Project/private/processed_test/leaderboard_metadata.json",
        help="DO NOT CHANGE. Path to leaderboard_metadata.json.",
    )
    parser.add_argument(
        "--features_dir",
        default="/projectnb/medaihack/VI_LUAD_Project/private/processed_test",
        help="DO NOT CHANGE. Directory containing pre-extracted test features.",
    )
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="Must match the hidden_dim used during training")
    parser.add_argument("--dropout", type=float, default=0.25,
                        help="Must match the dropout used during training")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
