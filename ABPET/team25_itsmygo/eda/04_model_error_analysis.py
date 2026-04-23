"""
04_model_error_analysis.py
==========================
EDA Script 4 — Model Error & Prediction Analysis

Justifies : post-training diagnostics — feeds back into ablations
            (loss δ tuning, per-tracer head adjustments, augmentation strength)
Reads     : val.csv, [predictions.csv]
Writes    : results/eda/04_model_error_analysis/

Two modes:
  (A) With model predictions → full error analysis
  (B) Without predictions    → naive baseline (mean predictor) analysis

Covers:
  - Predicted vs actual scatter (per tracer + colored by error)
  - Residual distribution & diagnostic plots (Q-Q, residual-vs-pred)
  - Bland-Altman agreement (per tracer)
  - Error stratified by amyloid status, tracer, and centiloid bin
  - Naive baselines (mean / median / per-tracer mean / per-tracer median)

Outputs:
    results/eda/04_model_error_analysis/
        ├── 01_predicted_vs_actual.png
        ├── 02_residual_distribution.png
        ├── 03_bland_altman.png
        ├── 04_error_by_centiloid_bin.png
        ├── 05_baseline_comparison.png
        └── error_report.txt

Usage:
    # With predictions:
    python eda/04_model_error_analysis.py \
        --val_csv  /projectnb/medaihack/ABPET/data/val.csv \
        --pred_csv results/predictions.csv

    # Baseline only (no model yet):
    python eda/04_model_error_analysis.py \
        --val_csv  /projectnb/medaihack/ABPET/data/val.csv
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats

# Shared EDA constants & style
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (
    AMYLOID_POS_THRESHOLD, TRACER_ORDER, TRACER_COLORS,
    setup_style, set_run_tag, log,
)

setup_style()


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_and_merge(val_csv: str, pred_csv: str | None) -> pd.DataFrame:
    val = pd.read_csv(val_csv)

    if pred_csv and os.path.exists(pred_csv):
        pred = pd.read_csv(pred_csv)
        # Merge on ID or npy_path
        merge_key = "ID" if "ID" in pred.columns else "npy_path"
        df = val.merge(pred[[merge_key, "PREDICTED_CENTILOIDS"]],
                       on=merge_key, how="left")
        print(f"  [✓] Loaded predictions from {pred_csv}")
    else:
        # Naive baseline: overall mean predictor
        mean_pred = val["CENTILOIDS"].mean()
        df = val.copy()
        df["PREDICTED_CENTILOIDS"] = mean_pred
        print(f"  [!] No predictions file found. Using naive mean predictor "
              f"(μ={mean_pred:.2f} CL)")

    df["residual"] = df["PREDICTED_CENTILOIDS"] - df["CENTILOIDS"]
    df["abs_error"] = df["residual"].abs()
    df["amyloid_status"] = np.where(
        df["CENTILOIDS"] >= AMYLOID_POS_THRESHOLD,
        "Positive", "Negative")
    df["centiloid_bin"] = pd.cut(
        df["CENTILOIDS"],
        bins=[-np.inf, 0, 25, 50, 75, 100, np.inf],
        labels=["<0", "0–25", "25–50", "50–75", "75–100", ">100"]
    )
    return df


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(df: pd.DataFrame, label: str = "ALL") -> dict:
    y_true = df["CENTILOIDS"]
    y_pred = df["PREDICTED_CENTILOIDS"]
    mae    = df["abs_error"].mean()
    rmse   = np.sqrt(((y_pred - y_true) ** 2).mean())
    r, _   = stats.pearsonr(y_true, y_pred)
    bias   = df["residual"].mean()
    return dict(label=label, n=len(df),
                MAE=mae, RMSE=rmse, r=r, bias=bias)


def print_metrics_table(df: pd.DataFrame):
    print(f"\n{'='*65}")
    print(f"  Model Error Metrics")
    print(f"{'='*65}")
    print(f"  {'Subset':<20} {'N':>5} {'MAE':>8} {'RMSE':>8} "
          f"{'r':>7} {'Bias':>8}")
    print(f"  {'-'*60}")

    # Overall
    m = compute_metrics(df, "ALL")
    print(f"  {'ALL':<20} {m['n']:>5} {m['MAE']:>8.2f} "
          f"{m['RMSE']:>8.2f} {m['r']:>7.3f} {m['bias']:>8.2f}")

    # Per tracer
    for tracer in TRACER_ORDER:
        sub = df[df["TRACER.AMY"] == tracer]
        if len(sub) < 2:
            continue
        m = compute_metrics(sub, tracer)
        print(f"  {tracer:<20} {m['n']:>5} {m['MAE']:>8.2f} "
              f"{m['RMSE']:>8.2f} {m['r']:>7.3f} {m['bias']:>8.2f}")

    # Per amyloid status
    for status in ["Positive", "Negative"]:
        sub = df[df["amyloid_status"] == status]
        if len(sub) < 2:
            continue
        m = compute_metrics(sub, f"Amyloid {status}")
        print(f"  {f'Amyloid {status}':<20} {m['n']:>5} {m['MAE']:>8.2f} "
              f"{m['RMSE']:>8.2f} {m['r']:>7.3f} {m['bias']:>8.2f}")

    print(f"{'='*65}")


# ── Plot Functions ─────────────────────────────────────────────────────────────

def plot_predicted_vs_actual(df: pd.DataFrame, out_dir: str):
    """Scatter plot of predictions vs ground truth, per tracer."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Predicted vs Actual Centiloid Score",
                 fontsize=15, fontweight="bold")

    lim = [min(df["CENTILOIDS"].min(), df["PREDICTED_CENTILOIDS"].min()) - 5,
           max(df["CENTILOIDS"].max(), df["PREDICTED_CENTILOIDS"].max()) + 5]

    # Shared fit line (same underlying data, so identical in both panels)
    m, b  = np.polyfit(df["CENTILOIDS"], df["PREDICTED_CENTILOIDS"], 1)
    r, _  = stats.pearsonr(df["CENTILOIDS"], df["PREDICTED_CENTILOIDS"])
    mae   = df["abs_error"].mean()
    x_fit = np.linspace(*lim, 200)

    for ax, color_by in zip(axes, ["tracer", "error"]):
        tracer_handles = []
        if color_by == "tracer":
            for tracer in TRACER_ORDER:
                sub = df[df["TRACER.AMY"] == tracer]
                h = ax.scatter(sub["CENTILOIDS"], sub["PREDICTED_CENTILOIDS"],
                               color=TRACER_COLORS[tracer], alpha=0.55,
                               s=25, label=tracer)
                tracer_handles.append(h)
            ax.set_title("Colored by Tracer")
        else:
            sc = ax.scatter(df["CENTILOIDS"], df["PREDICTED_CENTILOIDS"],
                            c=df["abs_error"], cmap="YlOrRd",
                            alpha=0.65, s=25, vmin=0,
                            vmax=df["abs_error"].quantile(0.95))
            plt.colorbar(sc, ax=ax, label="Absolute Error (CL)")
            ax.set_title("Colored by Absolute Error")

        # Reference lines (NOT added to the legend — annotated inline instead)
        ax.plot(lim, lim, "k--", lw=1.8)
        ax.plot(x_fit, m * x_fit + b, "b-", lw=1.5, alpha=0.7)

        # Single stats box: fit equation, Pearson r, MAE, n — placed at the
        # bottom-right so it never collides with the tracer legend (top-left).
        stats_txt = (f"y = x (perfect)\n"
                     f"Fit: y = {m:.2f}x + {b:.2f}\n"
                     f"Pearson r = {r:.3f}\n"
                     f"MAE = {mae:.2f} CL\n"
                     f"n = {len(df)}")
        ax.text(0.97, 0.03, stats_txt,
                transform=ax.transAxes, va="bottom", ha="right", fontsize=10,
                bbox=dict(fc="white", ec="gray", alpha=0.9))

        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("Actual Centiloid (CL)")
        ax.set_ylabel("Predicted Centiloid (CL)")
        ax.set_aspect("equal")

        # Legend only on the tracer panel, and only for tracer markers.
        if tracer_handles:
            ax.legend(handles=tracer_handles, title="Tracer",
                      fontsize=9, loc="upper left", framealpha=0.9)

    plt.tight_layout()
    path = os.path.join(out_dir, "01_predicted_vs_actual.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


def plot_residual_distribution(df: pd.DataFrame, out_dir: str):
    """Residual diagnostic plots."""
    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)
    fig.suptitle("Residual Diagnostics (Predicted − Actual)",
                 fontsize=15, fontweight="bold")

    residuals = df["residual"]

    # 1. Residual histogram + KDE
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(residuals, bins=35, density=True, alpha=0.6,
            color="#5B9BD5", edgecolor="white")
    kde_x = np.linspace(residuals.min()-5, residuals.max()+5, 300)
    ax.plot(kde_x, stats.gaussian_kde(residuals)(kde_x), "b-", lw=2)
    ax.axvline(0, color="red", lw=2, ls="--", label="Zero error")
    ax.axvline(residuals.mean(), color="orange", lw=1.8, ls="-.",
               label=f"Bias={residuals.mean():.2f}")
    ax.set_xlabel("Residual (CL)")
    ax.set_ylabel("Density")
    ax.set_title("Residual Distribution")
    ax.legend(fontsize=9)

    # 2. Q-Q plot
    ax = fig.add_subplot(gs[0, 1])
    (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
    ax.scatter(osm, osr, alpha=0.4, s=15, color="#5B9BD5")
    ax.plot(osm, slope * np.array(osm) + intercept,
            "r-", lw=2, label=f"Normal fit (r={r:.3f})")
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Sample Quantiles")
    ax.set_title("Q-Q Plot (Normality Check)")
    ax.legend()

    # 3. Residual vs predicted
    ax = fig.add_subplot(gs[0, 2])
    ax.scatter(df["PREDICTED_CENTILOIDS"], residuals,
               alpha=0.4, s=15, color="#5B9BD5")
    ax.axhline(0, color="red", lw=1.8, ls="--")
    ax.axhline(residuals.mean() + 1.96*residuals.std(),
               color="orange", lw=1.2, ls=":", label="±1.96 SD")
    ax.axhline(residuals.mean() - 1.96*residuals.std(),
               color="orange", lw=1.2, ls=":")
    ax.set_xlabel("Predicted Centiloid (CL)")
    ax.set_ylabel("Residual (CL)")
    ax.set_title("Residuals vs Predicted")
    ax.legend()

    # 4. Residuals per tracer (violin)
    ax = fig.add_subplot(gs[1, 0])
    palette = {t: TRACER_COLORS[t] for t in TRACER_ORDER}
    sns.violinplot(data=df, x="TRACER.AMY", y="residual",
                   hue="TRACER.AMY", hue_order=TRACER_ORDER,
                   order=TRACER_ORDER, palette=palette, ax=ax,
                   inner="box", cut=0, alpha=0.8, legend=False)
    ax.axhline(0, color="red", lw=1.8, ls="--")
    ax.set_title("Residuals per Tracer")
    ax.set_xlabel("Tracer")
    ax.set_ylabel("Residual (CL)")

    # 5. Absolute error per tracer
    ax = fig.add_subplot(gs[1, 1])
    sns.boxplot(data=df, x="TRACER.AMY", y="abs_error",
                hue="TRACER.AMY", hue_order=TRACER_ORDER,
                order=TRACER_ORDER, palette=palette, ax=ax,
                width=0.5, fliersize=3, legend=False)
    # Add MAE text
    for i, tracer in enumerate(TRACER_ORDER):
        sub = df[df["TRACER.AMY"] == tracer]
        mae = sub["abs_error"].mean()
        ax.text(i, sub["abs_error"].max() + 0.5,
                f"MAE\n{mae:.1f}", ha="center", fontsize=9)
    ax.set_title("Absolute Error per Tracer")
    ax.set_xlabel("Tracer")
    ax.set_ylabel("Absolute Error (CL)")

    # 6. Residuals per amyloid status
    ax = fig.add_subplot(gs[1, 2])
    sns.boxplot(data=df, x="amyloid_status", y="residual",
                hue="amyloid_status",
                palette={"Positive": "#F44336", "Negative": "#4CAF50"},
                ax=ax, width=0.4, fliersize=3, legend=False)
    ax.axhline(0, color="black", lw=1.8, ls="--")
    ax.set_title("Residuals by Amyloid Status")
    ax.set_xlabel("Amyloid Status")
    ax.set_ylabel("Residual (CL)")

    path = os.path.join(out_dir, "02_residual_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


def plot_bland_altman(df: pd.DataFrame, out_dir: str):
    """Bland-Altman agreement plot per tracer."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Bland-Altman Agreement Plot (per Tracer)",
                 fontsize=15, fontweight="bold")
    axes = axes.flatten()

    for i, tracer in enumerate(TRACER_ORDER):
        ax  = axes[i]
        sub = df[df["TRACER.AMY"] == tracer]

        mean_vals = (sub["CENTILOIDS"] + sub["PREDICTED_CENTILOIDS"]) / 2
        diff_vals = sub["PREDICTED_CENTILOIDS"] - sub["CENTILOIDS"]

        mean_diff = diff_vals.mean()
        std_diff  = diff_vals.std()
        loa_upper = mean_diff + 1.96 * std_diff
        loa_lower = mean_diff - 1.96 * std_diff

        ax.scatter(mean_vals, diff_vals,
                   color=TRACER_COLORS[tracer], alpha=0.55, s=20)
        ax.axhline(mean_diff,  color="red",    lw=2,   ls="--",
                   label=f"Bias={mean_diff:.2f}")
        ax.axhline(loa_upper,  color="orange", lw=1.5, ls=":",
                   label=f"+1.96SD={loa_upper:.2f}")
        ax.axhline(loa_lower,  color="orange", lw=1.5, ls=":",
                   label=f"−1.96SD={loa_lower:.2f}")
        ax.axhline(0, color="black", lw=1, ls="-", alpha=0.3)

        ax.set_title(f"{tracer} — Bland-Altman")
        ax.set_xlabel("Mean of Actual & Predicted (CL)")
        ax.set_ylabel("Predicted − Actual (CL)")
        ax.legend(fontsize=8)
        ax.text(0.02, 0.98,
                f"n={len(sub)}  MAE={sub['abs_error'].mean():.2f}",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(fc="white", ec="gray", alpha=0.7))

    plt.tight_layout()
    path = os.path.join(out_dir, "03_bland_altman.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


def plot_error_by_centiloid_bin(df: pd.DataFrame, out_dir: str):
    """Error analysis stratified by centiloid range."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.suptitle("Error by Centiloid Range",
                 fontsize=15, fontweight="bold")

    # MAE per bin
    ax = axes[0]
    bin_mae = df.groupby("centiloid_bin", observed=True)["abs_error"].mean()
    bin_n   = df.groupby("centiloid_bin", observed=True).size()
    bars = ax.bar(range(len(bin_mae)), bin_mae.values,
                  color=sns.color_palette("Blues_d", len(bin_mae)),
                  edgecolor="white", alpha=0.85)
    for bar, (b, n) in zip(bars, bin_n.items()):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.2,
                f"n={n}", ha="center", fontsize=9)
    ax.set_xticks(range(len(bin_mae)))
    ax.set_xticklabels(bin_mae.index, rotation=30)
    ax.set_xlabel("Centiloid Range")
    ax.set_ylabel("Mean Absolute Error (CL)")
    ax.set_title("MAE per Centiloid Bin")

    # Residual boxplot per bin
    ax = axes[1]
    bin_order = df["centiloid_bin"].cat.categories.tolist()
    sns.boxplot(data=df, x="centiloid_bin", y="residual",
                hue="centiloid_bin",
                order=bin_order, ax=ax,
                palette=sns.color_palette("Blues_d", len(bin_order)),
                width=0.5, fliersize=3, legend=False)
    ax.axhline(0, color="red", lw=1.8, ls="--")
    ax.set_xticklabels(bin_order, rotation=30)
    ax.set_xlabel("Centiloid Range")
    ax.set_ylabel("Residual (CL)")
    ax.set_title("Residual Boxplot per Bin")

    # Sample count + MAE heatmap by tracer x bin
    ax = axes[2]
    heatmap_data = df.pivot_table(
        values="abs_error", index="TRACER.AMY",
        columns="centiloid_bin", aggfunc="mean", observed=True
    ).reindex(index=TRACER_ORDER)
    sns.heatmap(heatmap_data, annot=True, fmt=".1f",
                cmap="YlOrRd", ax=ax, linewidths=0.5,
                cbar_kws={"label": "MAE (CL)"},
                mask=heatmap_data.isna())
    ax.set_title("MAE Heatmap (Tracer × CL Bin)")
    ax.set_xlabel("Centiloid Range")
    ax.set_ylabel("Tracer")

    plt.tight_layout()
    path = os.path.join(out_dir, "04_error_by_centiloid_bin.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


def plot_baseline_comparison(df: pd.DataFrame, out_dir: str):
    """Compare model vs naive baselines (mean / median / tracer-mean)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle("Model vs Naive Baseline Comparison",
                 fontsize=15, fontweight="bold")

    cl = df["CENTILOIDS"]

    # Build baseline predictions
    baselines = {
        "Global Mean"  : np.full(len(df), cl.mean()),
        "Global Median": np.full(len(df), cl.median()),
        "Tracer Mean"  : df["TRACER.AMY"].map(
            df.groupby("TRACER.AMY")["CENTILOIDS"].mean()).values,
        "Tracer Median": df["TRACER.AMY"].map(
            df.groupby("TRACER.AMY")["CENTILOIDS"].median()).values,
    }
    if "PREDICTED_CENTILOIDS" in df.columns:
        baselines["Model"] = df["PREDICTED_CENTILOIDS"].values

    # Compute metrics
    mae_vals: dict[str, float] = {}
    r_vals:   dict[str, float] = {}
    for name, preds in baselines.items():
        mae_vals[name] = float(np.abs(preds - cl.values).mean())
        # A constant predictor has undefined Pearson r → report NaN, not a fake 1.0
        if preds.std() == 0:
            r_vals[name] = float("nan")
        else:
            r, _ = stats.pearsonr(cl.values, preds)
            r_vals[name] = float(r)

    if "Model" in baselines:
        colors = ["#90A4AE"] * (len(baselines) - 1) + ["#1976D2"]
    else:
        colors = ["#90A4AE"] * len(baselines)

    # MAE bar
    ax = axes[0]
    bars = ax.bar(mae_vals.keys(), mae_vals.values(),
                  color=colors, edgecolor="white", alpha=0.85)
    for bar, v in zip(bars, mae_vals.values()):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.3,
                f"{v:.2f}", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("MAE (CL)")
    ax.set_title("Mean Absolute Error ↓")
    ax.set_xticklabels(mae_vals.keys(), rotation=20, ha="right")
    # Baseline reference line
    ax.axhline(list(mae_vals.values())[0],
               color="red", lw=1.5, ls="--", alpha=0.5, label="Global Mean")

    # Pearson r bar
    ax = axes[1]
    bars = ax.bar(r_vals.keys(), r_vals.values(),
                  color=colors, edgecolor="white", alpha=0.85)
    for bar, v in zip(bars, r_vals.values()):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("Pearson r")
    ax.set_title("Pearson Correlation ↑")
    ax.set_ylim(0, 1.05)
    ax.set_xticklabels(r_vals.keys(), rotation=20, ha="right")

    plt.tight_layout()
    path = os.path.join(out_dir, "05_baseline_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


def save_error_report(df: pd.DataFrame, pred_csv: str | None, out_dir: str):
    lines = [
        "Model Error Analysis Report",
        "=" * 60,
        f"Predictions source: {pred_csv if pred_csv else 'Naive mean baseline'}",
        "",
        f"{'Subset':<22} {'N':>5} {'MAE':>8} {'RMSE':>8} {'r':>7} {'Bias':>8}",
        "-" * 60,
    ]

    for label, sub in [("ALL", df)] + \
                       [(t, df[df["TRACER.AMY"] == t]) for t in TRACER_ORDER] + \
                       [("Amyloid Positive",
                         df[df["amyloid_status"] == "Positive"]),
                        ("Amyloid Negative",
                         df[df["amyloid_status"] == "Negative"])]:
        if len(sub) < 2:
            continue
        m = compute_metrics(sub, label)
        lines.append(
            f"  {label:<20} {m['n']:>5} {m['MAE']:>8.2f} "
            f"{m['RMSE']:>8.2f} {m['r']:>7.3f} {m['bias']:>8.2f}"
        )

    # Normality test on residuals
    _, p_norm = stats.shapiro(df["residual"].sample(
        min(300, len(df)), random_state=42))
    lines += [
        "=" * 60,
        f"Residual Normality (Shapiro-Wilk, n≤300): p={p_norm:.4f}",
        f"  → {'Approximately normal' if p_norm > 0.05 else 'Non-normal residuals (p≤0.05)'}",
        f"Residual Bias: {df['residual'].mean():.4f} CL",
        f"Residual Std : {df['residual'].std():.4f} CL",
    ]

    path = os.path.join(out_dir, "error_report.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved → {path}")
    print("\n" + "\n".join(lines))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="EDA 04 — Model Error Analysis")
    parser.add_argument("--val_csv",  default="data/val.csv")
    parser.add_argument("--pred_csv", default=None,
                        help="Path to predictions.csv "
                             "(columns: ID, PREDICTED_CENTILOIDS)")
    parser.add_argument("--out_dir",  default="results/eda/post_train/04_model_error_analysis")
    parser.add_argument("--tag",      default=None,
                        help="Label stamped on every figure "
                             "(e.g. run/experiment name). "
                             "Defaults to basename(pred_csv) or basename(out_dir).")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    # For post-train analysis the pred_csv identifies the model/run; fall back
    # to the output folder name otherwise (e.g. naive-baseline invocations).
    default_tag = (
        os.path.splitext(os.path.basename(args.pred_csv))[0]
        if args.pred_csv else os.path.basename(args.out_dir.rstrip("/\\"))
    )
    set_run_tag(args.tag or default_tag)
    print(f"\n{'#'*55}")
    print("  EDA 04 — Model Error Analysis")
    print(f"  Val CSV  → {args.val_csv}")
    print(f"  Pred CSV → {args.pred_csv or 'None (using naive baseline)'}")
    print(f"  Output   → {args.out_dir}")
    print(f"{'#'*55}")

    df = load_and_merge(args.val_csv, args.pred_csv)
    print_metrics_table(df)

    print("\n[+] Generating plots...")
    plot_predicted_vs_actual(df, args.out_dir)
    plot_residual_distribution(df, args.out_dir)
    plot_bland_altman(df, args.out_dir)
    plot_error_by_centiloid_bin(df, args.out_dir)
    plot_baseline_comparison(df, args.out_dir)
    save_error_report(df, args.pred_csv, args.out_dir)

    print(f"\n✓ All outputs saved to → {args.out_dir}\n")


if __name__ == "__main__":
    main()