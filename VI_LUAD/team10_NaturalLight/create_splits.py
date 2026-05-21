"""
create_splits.py — Patient-Level Cross-Validation Splits for VI-LUAD
=====================================================================
This script creates 5-fold stratified patient-level cross-validation splits
from the hackathon label file. Run this once before training — train_eval.py
reads the JSON files it produces.

Why patient-level splits?
--------------------------
The same patient may contribute multiple WSI slides. If we split at the slide
level, slides from the same patient could appear in both train and test — this
is data leakage that inflates performance. Patient-level splits ensure all
slides from a patient are in either train or test, never both.

Why stratified?
---------------
With imbalanced classes, random splits might put most VI-positive patients in
train and few in test. Stratification ensures each fold's train/test set has
the same class ratio as the full dataset.

Output (in --splits_dir, default: splits/):
  fold_0.json  ...  fold_4.json

Each JSON file contains:
  {
    "train": [{"pid": "...", "filename": "...", "vi_label": "..."}, ...],
    "test":  [{"pid": "...", "filename": "...", "vi_label": "..."}, ...]
  }

Usage:
  python create_splits.py

  # Custom splits directory or number of folds:
  python create_splits.py \\
      --splits_dir my_splits --n_folds 3 --random_seed 0

  # Print the label distribution and fold summary, but don't save:
  python create_splits.py --dry_run
"""

import os
import csv
import json
import argparse
from collections import defaultdict, Counter

import numpy as np
from sklearn.model_selection import StratifiedKFold


# =============================================================================
# SECTION 1: LABEL FILE PARSING
# =============================================================================

def load_label_file(label_file: str):
    """
    Parse hackathon_label.txt and return a list of slide records.

    The label file is tab-separated (R format) with columns:
      row_idx | pid | filename | vi_label | sp_label

    Only VITUMOR and NONVITUMOR slides are returned — NONTUMOR slides are
    excluded from classification splits (features are available for optional use).

    Parameters
    ----------
    label_file : path to hackathon_label.txt

    Returns
    -------
    slides : list of dicts with keys: pid, filename, vi_label
    """
    slides = []
    with open(label_file, newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # skip header row
        for row in reader:
            if len(row) < 4:
                continue
            _, pid, filename, vi_label = row[0], row[1], row[2], row[3]
            if vi_label not in ("VITUMOR", "NONVITUMOR"):
                continue  # skip NONTUMOR and any unexpected labels
            slides.append({"pid": pid, "filename": filename, "vi_label": vi_label})
    return slides


# =============================================================================
# SECTION 2: PATIENT-LEVEL STRATIFIED SPLITS
# =============================================================================

def assign_patient_label(slide_labels: list) -> str:
    """
    Assign a single VI label to a patient from their set of slide labels.

    A patient is VITUMOR if any of their slides shows vascular invasion;
    otherwise NONVITUMOR.
    """
    if "VITUMOR" in slide_labels:
        return "VITUMOR"
    return "NONVITUMOR"


def create_patient_splits(slides: list, n_folds: int = 5, random_seed: int = 42) -> list:
    """
    Create n-fold stratified cross-validation splits at the patient level.

    Parameters
    ----------
    slides      : list of slide dicts (from load_label_file)
    n_folds     : number of cross-validation folds (default: 5)
    random_seed : random seed for reproducibility (default: 42)

    Returns
    -------
    folds : list of n_folds dicts, each with:
              "train": list of {pid, filename, vi_label} for training slides
              "test" : list of {pid, filename, vi_label} for test slides

    Tips for improvement
    --------------------
    - Add a validation split: change "test" to "val" and add a held-out "test"
      set — useful for tuning hyperparameters without touching the true test set.
    - Adjust n_folds: fewer folds (e.g. 3) for faster iteration; more folds
      for a more reliable performance estimate.
    - Change random_seed: train across several seeds to check stability.
    """
    # Group slides by patient
    pid_to_slides = defaultdict(list)
    for s in slides:
        pid_to_slides[s["pid"]].append(s)

    # Assign patient-level labels for stratification
    pids = sorted(pid_to_slides.keys())
    pid_labels = [
        assign_patient_label([s["vi_label"] for s in pid_to_slides[p]])
        for p in pids
    ]

    print(f"\nPatient-level label distribution:")
    label_dist = Counter(pid_labels)
    for label, count in sorted(label_dist.items()):
        print(f"  {label}: {count} patients")
    print(f"  Total: {len(pids)} patients, {len(slides)} slides")

    # Cap n_folds if a class has fewer patients than requested folds
    min_class_count = min(label_dist.values())
    if n_folds > min_class_count:
        n_folds = min_class_count
        print(f"\n  Note: n_folds reduced to {n_folds} "
              f"(smallest class has {min_class_count} patients)")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    pids_arr   = np.array(pids)
    labels_arr = np.array(pid_labels)

    folds = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(pids_arr, labels_arr)):
        train_pids = set(pids_arr[train_idx])
        test_pids  = set(pids_arr[test_idx])

        train_slides = [s for s in slides if s["pid"] in train_pids]
        test_slides  = [s for s in slides if s["pid"] in test_pids]

        folds.append({"train": train_slides, "test": test_slides})

        train_labels = Counter(s["vi_label"] for s in train_slides)
        test_labels  = Counter(s["vi_label"] for s in test_slides)
        print(f"\nFold {fold_idx}: "
              f"train={len(train_slides)} slides ({len(train_pids)} patients) | "
              f"test={len(test_slides)} slides ({len(test_pids)} patients)")
        print(f"  train labels: {dict(train_labels)}")
        print(f"  test  labels: {dict(test_labels)}")

    return folds


# =============================================================================
# SECTION 3: SAVE SPLITS
# =============================================================================

def save_splits(folds: list, splits_dir: str):
    """
    Save cross-validation splits to JSON files.

    Each file (fold_0.json ... fold_4.json) contains:
      {
        "train": [{"pid": ..., "filename": ..., "vi_label": ...}, ...],
        "test":  [{"pid": ..., "filename": ..., "vi_label": ...}, ...]
      }
    """
    os.makedirs(splits_dir, exist_ok=True)
    for i, fold in enumerate(folds):
        out_path = os.path.join(splits_dir, f"fold_{i}.json")
        with open(out_path, "w") as f:
            json.dump(fold, f, indent=2)
        print(f"Saved fold {i} → {out_path}")


# =============================================================================
# SECTION 4: ENTRY POINT
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Create patient-level stratified cross-validation splits for VI-LUAD",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--label_file",
        type=str,
        default="/projectnb/medaihack/VI_LUAD_Project/Clinical_Data/hackathon_label.txt",
        help="Path to hackathon_label.txt",
    )
    parser.add_argument(
        "--splits_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "splits"),
        help="Directory to save fold_0.json ... fold_N.json",
    )
    parser.add_argument(
        "--n_folds",
        type=int,
        default=5,
        help="Number of cross-validation folds",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the label distribution and fold summary without saving any files",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("=" * 60)
    print("STEP: Loading label file")
    print("=" * 60)
    slides = load_label_file(args.label_file)
    print(f"Loaded {len(slides)} VITUMOR/NONVITUMOR slides")

    print("\n" + "=" * 60)
    print("STEP: Creating cross-validation splits")
    print("=" * 60)
    folds = create_patient_splits(slides, n_folds=args.n_folds,
                                  random_seed=args.random_seed)

    if args.dry_run:
        print("\nDry run — no files written.")
    else:
        print("\n" + "=" * 60)
        print("STEP: Saving splits")
        print("=" * 60)
        save_splits(folds, args.splits_dir)
        print(f"\nDone! Splits saved to: {args.splits_dir}")
