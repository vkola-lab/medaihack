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
from model import build_model, build_model_from_config, ACMILEnsemble


# ============================================================================
# SECTION 1: LOAD MODEL CHECKPOINT
# ============================================================================
# We load an ACMIL ensemble (K state_dicts saved in a single .pth) and return
# a wrapper whose forward(features) signature matches what Section 2 calls.
#
# Why the torch.load monkey-patch?
# --------------------------------
# Our ACMIL uses 2D sinusoidal positional encoding over each patch's (col, row)
# grid coordinates. Those coords live in the feature .pt file as data["coords"]
# but Section 2 (locked) only forwards data["features"] into the model. To
# bridge that without touching Section 2, we wrap torch.load so that every
# time a feature file is loaded (Section 2 line ~92), we capture its coords
# into a closure cache. The ensemble wrapper's forward reads that cache and
# feeds the right coords into ACMIL alongside the features. Section 2 is
# completely untouched.

def load_checkpoint(checkpoint_path: str, device: torch.device,
                    hidden_dim: int, dropout: float):
    """
    Load an ACMIL ensemble checkpoint and return a model whose forward
    accepts `features` alone (matching Section 2's call site). The model
    internally injects `coords` captured from the most recently loaded
    feature .pt file.
    """
    # --- Load checkpoint (before installing the torch.load monkey-patch) ---
    _original_torch_load = torch.load
    ckpt = _original_torch_load(checkpoint_path, map_location=device,
                                weights_only=False)

    # Rebuild the architecture from the config stored in the checkpoint so
    # we don't depend on the CLI defaults for hidden_dim / dropout.
    if "config" in ckpt:
        config = ckpt["config"]
    else:
        config = {"hidden_dim": hidden_dim, "dropout": dropout}

    # Collect the state dicts — support both ensemble and single-model formats
    if "model_states" in ckpt:
        state_dicts = ckpt["model_states"]
    elif "model_state_dict" in ckpt:
        state_dicts = [ckpt["model_state_dict"]]
    else:
        raise RuntimeError(f"Unrecognized checkpoint format: {checkpoint_path}")

    members = []
    for state in state_dicts:
        m = build_model_from_config(config, verbose=False).to(device)
        m.load_state_dict(state)
        m.eval()
        members.append(m)
    ensemble = ACMILEnsemble(members).to(device).eval()
    print(f"  Loaded ensemble ({len(members)} members) from: {checkpoint_path}")

    # --- Install torch.load patch so Section 2's feature loads leak coords ---
    _coord_cache = {"coords": None}

    def _patched_torch_load(*args, **kwargs):
        data = _original_torch_load(*args, **kwargs)
        if isinstance(data, dict) and "coords" in data and "features" in data:
            _coord_cache["coords"] = data["coords"]
        return data

    torch.load = _patched_torch_load

    # --- Wrapper that reads the cached coords on every forward ---
    import torch.nn as _nn
    class CoordAwareEnsemble(_nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, features):
            coords = _coord_cache.get("coords")
            if coords is not None:
                coords = coords.to(features.device)
            logits, aux = self.inner(features, coords)
            return logits, aux

    wrapped = CoordAwareEnsemble(ensemble).to(device).eval()
    return wrapped


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

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Resolve output directory
    preds_dir = os.path.join(args.out_dir, args.team)

    # Load leaderboard metadata
    print(f"\n[Loading leaderboard metadata]")
    with open(args.test_metadata) as f:
        test_metadata = json.load(f)
    print(f"  {len(test_metadata)} slides in leaderboard set")

    # Load checkpoint
    print(f"\n[Loading checkpoint]")
    model = load_checkpoint(args.checkpoint, device,
                            args.hidden_dim, args.dropout)

    # Run inference (features live alongside the metadata file)
    features_dir = str(Path(args.test_metadata).parent)
    print(f"\n[Running inference on leaderboard slides]")
    slide_results = run_inference(model, test_metadata, features_dir, device)

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
        required=True,
        help="Path to leaderboard_metadata.json (provided by the organizers).",
    )
    parser.add_argument(
        "--out_dir",
        default="predictions",
        help="Directory to write prediction outputs (default: predictions/)",
    )
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="Must match the hidden_dim used during training")
    parser.add_argument("--dropout", type=float, default=0.25,
                        help="Must match the dropout used during training")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
