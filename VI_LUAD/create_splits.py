"""
create_splits.py — Patient-Level CV Splits with Val Carve-Out
=============================================================
Creates patient-level stratified splits from the hackathon label file.

Output layout (in --splits_dir, default: starter_code/splits/):
  fold_0.json ... fold_{N-1}.json   — each {"train": [...], "val": [...], "test": [...]}

5-fold CV with a validation carve-out.
Each fold has its own train / val / test. The val split is used for
early stopping and temperature scaling.

Usage:
  python starter_code/create_splits.py
  python starter_code/create_splits.py --val_frac 0.15
  python starter_code/create_splits.py --n_folds 5 --val_frac 0.15 --dry_run
"""

import os
import csv
import json
import argparse
from collections import defaultdict, Counter

import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


# =============================================================================
# SECTION 1: LABEL FILE PARSING
# =============================================================================

def load_label_file(label_file: str):
    """Parse hackathon_label.txt. Only VITUMOR / NONVITUMOR rows are kept."""
    slides = []
    with open(label_file, newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # header
        for row in reader:
            if len(row) < 4:
                continue
            _, pid, filename, vi_label = row[0], row[1], row[2], row[3]
            if vi_label not in ("VITUMOR", "NONVITUMOR"):
                continue
            slides.append({"pid": pid, "filename": filename, "vi_label": vi_label})
    return slides


def load_nontumor_slides(label_file: str):
    """
    Parse hackathon_label.txt for NONTUMOR rows. These slides have no tumor,
    so vascular invasion is impossible → they act as guaranteed negatives.
    Each NONTUMOR slide is wrapped as its own synthetic NONVITUMOR patient
    (pid = "NONTUMOR__<filename>") so that our patient-level max-aggregation
    loss treats it as an independent negative example.
    """
    slides = []
    with open(label_file, newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # header
        for row in reader:
            if len(row) < 4:
                continue
            _, pid, filename, vi_label = row[0], row[1], row[2], row[3]
            if vi_label != "NONTUMOR":
                continue
            slides.append({
                "pid":      f"NONTUMOR__{filename}",
                "filename": filename,
                "vi_label": "NONVITUMOR",  # no tumor ⇒ no VI
            })
    return slides


# =============================================================================
# SECTION 2: PATIENT-LEVEL STRATIFIED SPLITS
# =============================================================================

def assign_patient_label(slide_labels):
    return "VITUMOR" if "VITUMOR" in slide_labels else "NONVITUMOR"


def group_by_patient(slides):
    pid_to_slides = defaultdict(list)
    for s in slides:
        pid_to_slides[s["pid"]].append(s)
    pids = sorted(pid_to_slides.keys())
    pid_labels = [
        assign_patient_label([s["vi_label"] for s in pid_to_slides[p]])
        for p in pids
    ]
    return pids, pid_labels, pid_to_slides


def slices_of(slides_list, pid_set):
    return [s for s in slides_list if s["pid"] in pid_set]


def create_splits(slides,
                  n_folds: int = 5,
                  val_frac: float = 0.15,
                  random_seed: int = 42,
                  extra_train_negatives=None):
    """
    Returns
    -------
    folds : list of dicts with keys "train", "val", "test" (lists of slide dicts)
    """
    pids, pid_labels, pid_to_slides = group_by_patient(slides)

    print(f"\nPatient-level label distribution:")
    label_dist = Counter(pid_labels)
    for label, count in sorted(label_dist.items()):
        print(f"  {label}: {count} patients")
    print(f"  Total: {len(pids)} patients, {len(slides)} slides")

    # Cap n_folds by smallest class
    label_counts = Counter(pid_labels)
    min_class = min(label_counts.values())
    if n_folds > min_class:
        n_folds = min_class
        print(f"\n  Note: n_folds reduced to {n_folds} "
              f"(smallest class has {min_class} patients)")

    # 5-fold CV
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    pids_arr   = np.array(pids)
    labels_arr = np.array(pid_labels)

    folds = []
    for fold_idx, (trainval_idx, test_idx) in enumerate(
            skf.split(pids_arr, labels_arr)):
        trainval_pids_arr   = pids_arr[trainval_idx]
        trainval_labels_arr = labels_arr[trainval_idx]
        test_pids_set       = set(pids_arr[test_idx])

        # Stratified val carve-out from each fold's train
        if val_frac > 0.0:
            val_splitter = StratifiedShuffleSplit(
                n_splits=1, test_size=val_frac,
                random_state=random_seed + fold_idx + 1)
            (tr_idx, v_idx), = val_splitter.split(
                trainval_pids_arr, trainval_labels_arr)
            train_pids_set = set(trainval_pids_arr[tr_idx])
            val_pids_set   = set(trainval_pids_arr[v_idx])
        else:
            train_pids_set = set(trainval_pids_arr)
            val_pids_set   = set()

        train_slides = slices_of(slides, train_pids_set)
        val_slides   = slices_of(slides, val_pids_set)
        test_slides  = slices_of(slides, test_pids_set)

        # NONTUMOR augmentation: add guaranteed negatives to TRAIN ONLY.
        # Never leak them into val/test — their patient IDs are synthetic and
        # they shouldn't skew the calibration/selection signal.
        n_extra = 0
        if extra_train_negatives:
            train_slides = train_slides + list(extra_train_negatives)
            n_extra = len(extra_train_negatives)

        folds.append({
            "train": train_slides,
            "val":   val_slides,
            "test":  test_slides,
        })

        extra_str = f" (+{n_extra} NONTUMOR)" if n_extra else ""
        print(f"\nFold {fold_idx}: "
              f"train={len(train_slides)} ({len(train_pids_set)} pats){extra_str} | "
              f"val={len(val_slides)} ({len(val_pids_set)} pats) | "
              f"test={len(test_slides)} ({len(test_pids_set)} pats)")
        print(f"  train labels: {dict(Counter(s['vi_label'] for s in train_slides))}")
        print(f"  val   labels: {dict(Counter(s['vi_label'] for s in val_slides))}")
        print(f"  test  labels: {dict(Counter(s['vi_label'] for s in test_slides))}")

    return folds


# =============================================================================
# SECTION 3: SAVE
# =============================================================================

def save_splits(folds, splits_dir: str):
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
        description="Create patient-level train/val/test splits",
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
        help="Directory to save fold_*.json",
    )
    parser.add_argument("--n_folds",     type=int,   default=5)
    parser.add_argument("--val_frac",    type=float, default=0.15,
                        help="Fraction of each fold's train+val pool to use as val.")
    parser.add_argument("--include_nontumor", action="store_true",
                        help="Add the 203 NONTUMOR slides to each fold's TRAIN split "
                             "as extra guaranteed-negative examples (never in val/test).")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true",
                        help="Print splits but don't write any files.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("=" * 60)
    print("STEP: Loading label file")
    print("=" * 60)
    slides = load_label_file(args.label_file)
    print(f"Loaded {len(slides)} VITUMOR/NONVITUMOR slides")

    extra_neg = None
    if args.include_nontumor:
        extra_neg = load_nontumor_slides(args.label_file)
        print(f"Loaded {len(extra_neg)} NONTUMOR slides "
              f"(will be added to each fold's TRAIN split only)")

    print("\n" + "=" * 60)
    print("STEP: Creating cross-validation splits")
    print("=" * 60)
    folds = create_splits(
        slides,
        n_folds=args.n_folds,
        val_frac=args.val_frac,
        random_seed=args.random_seed,
        extra_train_negatives=extra_neg,
    )

    if args.dry_run:
        print("\nDry run — no files written.")
    else:
        print("\n" + "=" * 60)
        print("STEP: Saving splits")
        print("=" * 60)
        save_splits(folds, args.splits_dir)
        print(f"\nDone! Splits saved to: {args.splits_dir}")
