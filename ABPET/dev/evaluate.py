"""
evaluate.py

Compare predictions.csv against val.csv (ground truth) and report MSE.

Usage:
    python evaluate.py \
        --pred  predictions.csv \
        --gt    /projectnb/medaihack/ABPET/data/val.csv
"""

import argparse
import os, sys
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mygo_centiloid.utils import (
    make_run_dir, write_config, write_metrics, append_registry,
)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))

def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    r, _ = pearsonr(y_true, y_pred)
    return float(r)


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def print_report(label: str, y_true: np.ndarray, y_pred: np.ndarray, n: int):
    mse   = mean_squared_error(y_true, y_pred)
    rmse  = root_mean_squared_error(y_true, y_pred)
    mae   = mean_absolute_error(y_true, y_pred)
    r     = pearson_r(y_true, y_pred)

    print(f"  {'Tracer':<10} {'N':>5} {'MSE':>10} {'RMSE':>10} {'MAE':>10} {'Pearson r':>10}")
    print(f"  {'-'*57}")
    print(f"  {label:<10} {n:>5} {mse:>10.4f} {rmse:>10.4f} {mae:>10.4f} {r:>10.4f}")
    return mse


def evaluate(pred_path: str, gt_path: str):

    # ── Load files ────────────────────────────────────────────────────────
    pred_df = pd.read_csv(pred_path)
    gt_df   = pd.read_csv(gt_path)

    # ── Validate required columns ─────────────────────────────────────────
    required_pred = {"ID", "PREDICTED_CENTILOIDS"}
    required_gt   = {"ID", "CENTILOIDS", "TRACER.AMY"}

    missing_pred = required_pred - set(pred_df.columns)
    missing_gt   = required_gt   - set(gt_df.columns)

    if missing_pred:
        raise ValueError(f"predictions.csv is missing columns: {missing_pred}")
    if missing_gt:
        raise ValueError(f"val.csv is missing columns: {missing_gt}")

    # ── Merge on ID ───────────────────────────────────────────────────────
    merged = pd.merge(gt_df[["ID", "CENTILOIDS", "TRACER.AMY"]],
                      pred_df[["ID", "PREDICTED_CENTILOIDS"]],
                      on="ID", how="inner")

    n_gt   = len(gt_df)
    n_pred = len(pred_df)
    n_merged = len(merged)

    print("\n" + "=" * 57)
    print("  AMYLOID PET — CENTILOID PREDICTION EVALUATION")
    print("=" * 57)
    print(f"  Ground truth samples : {n_gt}")
    print(f"  Predicted samples    : {n_pred}")
    print(f"  Matched samples      : {n_merged}")

    if n_merged == 0:
        raise ValueError("No matching IDs found between pred and gt files.")
    if n_merged < n_gt:
        print(f"  ⚠️  WARNING: {n_gt - n_merged} ground truth IDs have no prediction!")

    print()

    y_true = merged["CENTILOIDS"].values.astype(np.float32)
    y_pred = merged["PREDICTED_CENTILOIDS"].values.astype(np.float32)

    # ── Overall metrics ───────────────────────────────────────────────────
    print("  ── OVERALL ──────────────────────────────────────────")
    overall_mse = print_report("ALL", y_true, y_pred, n_merged)
    print()

    metrics = {
        "ALL": {
            "n":    n_merged,
            "mse":  mean_squared_error(y_true, y_pred),
            "rmse": root_mean_squared_error(y_true, y_pred),
            "mae":  mean_absolute_error(y_true, y_pred),
            "r":    pearson_r(y_true, y_pred),
        }
    }

    # ── Per-tracer metrics ────────────────────────────────────────────────
    print("  ── PER TRACER ───────────────────────────────────────")
    print(f"  {'Tracer':<10} {'N':>5} {'MSE':>10} {'RMSE':>10} {'MAE':>10} {'Pearson r':>10}")
    print(f"  {'-'*57}")

    for tracer, group in merged.groupby("TRACER.AMY"):
        t_true = group["CENTILOIDS"].values.astype(np.float32)
        t_pred = group["PREDICTED_CENTILOIDS"].values.astype(np.float32)
        n      = len(group)

        mse  = mean_squared_error(t_true, t_pred)
        rmse = root_mean_squared_error(t_true, t_pred)
        mae  = mean_absolute_error(t_true, t_pred)
        r    = pearson_r(t_true, t_pred)

        print(f"  {tracer:<10} {n:>5} {mse:>10.4f} {rmse:>10.4f} {mae:>10.4f} {r:>10.4f}")
        metrics[str(tracer)] = {"n": n, "mse": mse, "rmse": rmse, "mae": mae, "r": r}

    print("=" * 57)
    print(f"\n  ✅  Overall MSE: {overall_mse:.4f}\n")

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate centiloid predictions.")
    parser.add_argument("--pred", type=str, required=True,
                        help="Path to predictions.csv (must have ID, PREDICTED_CENTILOIDS)")
    parser.add_argument("--gt",   type=str, required=True,
                        help="Path to val.csv (must have ID, CENTILOIDS, TRACER.AMY)")
    parser.add_argument("--model",    type=str, default="unknown",
                        help="Model name for registry bookkeeping "
                             "(e.g. petresnet, petresnet_no_film).")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--log_dir",  type=str, default="logs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_dir = make_run_dir(
        run_type="evaluate", model=args.model,
        base=args.log_dir, run_name=args.run_name,
    )
    write_config(run_dir, {"run_type": "evaluate", **vars(args)})
    print(f"Run dir: {run_dir}\n")

    metrics = evaluate(pred_path=args.pred, gt_path=args.gt)

    write_metrics(run_dir, {"run_type": "evaluate", "model": args.model,
                            "pred": args.pred, "gt": args.gt,
                            "subsets": metrics})
    append_registry({
        "type":    "evaluate",
        "model":   args.model,
        "run_dir": str(run_dir),
        "pred":    args.pred,
        "gt":      args.gt,
        "metrics": {"ALL": metrics["ALL"]},
    })