"""
01_centiloid_distribution.py
============================
EDA Script 1 — Centiloid Score Distribution Analysis

Justifies : HuberLoss(δ=25) [losses.py], WeightedRandomSampler [train.py]
Reads     : train.csv, val.csv  (CENTILOIDS column)
Writes    : results/eda/01_centiloid_distribution/

Analyzes the distribution of centiloid scores across:
- Overall train / val splits
- Amyloid-positive vs amyloid-negative groups (threshold = 24.4 CL)

Outputs:
    results/eda/01_centiloid_distribution/
        ├── 01_overall_distribution.png
        ├── 02_train_val_comparison.png
        ├── 03_amyloid_pos_neg.png
        ├── 04_cumulative_distribution.png
        └── summary_stats.txt
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
from _common import AMYLOID_POS_THRESHOLD, setup_style, set_run_tag, log

# ── Config ────────────────────────────────────────────────────────────────────
PALETTE = sns.color_palette("Set2")
setup_style()

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_data(train_csv: str, val_csv: str):
    train = pd.read_csv(train_csv)
    val   = pd.read_csv(val_csv)
    train["split"] = "train"
    val["split"]   = "val"
    combined = pd.concat([train, val], ignore_index=True)
    combined["amyloid_status"] = np.where(
        combined["CENTILOIDS"] >= AMYLOID_POS_THRESHOLD,
        "Positive (≥24.4 CL)", "Negative (<24.4 CL)"
    )
    return train, val, combined


def print_summary(df: pd.DataFrame, label: str):
    cl = df["CENTILOIDS"]
    print(f"\n{'='*50}")
    print(f"  {label}  (n={len(df)})")
    print(f"{'='*50}")
    print(f"  Mean   : {cl.mean():.2f} CL")
    print(f"  Median : {cl.median():.2f} CL")
    print(f"  Std    : {cl.std():.2f} CL")
    print(f"  Min    : {cl.min():.2f} CL")
    print(f"  Max    : {cl.max():.2f} CL")
    print(f"  Skew   : {cl.skew():.3f}")
    print(f"  Kurtosis: {cl.kurtosis():.3f}")
    pos_n = (cl >= AMYLOID_POS_THRESHOLD).sum()
    print(f"  Amyloid Positive: {pos_n} ({100*pos_n/len(df):.1f}%)")
    print(f"  Amyloid Negative: {len(df)-pos_n} ({100*(len(df)-pos_n)/len(df):.1f}%)")
    pcts = [10, 25, 50, 75, 90, 95]
    print(f"  Percentiles: " +
          ", ".join(f"P{p}={cl.quantile(p/100):.1f}" for p in pcts))


# ── Plot Functions ─────────────────────────────────────────────────────────────

def plot_overall_distribution(combined: pd.DataFrame, out_dir: str):
    """Histogram + KDE + rug plot of all centiloid scores."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Centiloid Score — Overall Distribution (Train + Val)",
                 fontsize=15, fontweight="bold")

    cl = combined["CENTILOIDS"]

    # --- Left: Histogram + KDE ---
    ax = axes[0]
    ax.hist(cl, bins=40, color=PALETTE[0], alpha=0.7,
            edgecolor="white", density=True, label="Histogram")
    kde_x = np.linspace(cl.min() - 5, cl.max() + 5, 300)
    kde   = stats.gaussian_kde(cl)
    ax.plot(kde_x, kde(kde_x), color=PALETTE[1], lw=2.5, label="KDE")
    ax.axvline(cl.mean(),   color="red",    lw=1.8, ls="--", label=f"Mean={cl.mean():.1f}")
    ax.axvline(cl.median(), color="orange", lw=1.8, ls="-.", label=f"Median={cl.median():.1f}")
    ax.axvline(AMYLOID_POS_THRESHOLD, color="black", lw=1.5,
               ls=":", label=f"Threshold={AMYLOID_POS_THRESHOLD} CL")
    ax.set_xlabel("Centiloid Score (CL)")
    ax.set_ylabel("Density")
    ax.set_title("Histogram + KDE")
    ax.legend(fontsize=10)

    # --- Right: Box + Swarm ---
    ax = axes[1]
    bp = ax.boxplot(cl, vert=True, patch_artist=True,
                    boxprops=dict(facecolor=PALETTE[0], alpha=0.7),
                    medianprops=dict(color="red", lw=2))
    # Jittered points (seeded for reproducibility)
    rng = np.random.default_rng(42)
    jitter = rng.normal(1, 0.04, len(cl))
    ax.scatter(jitter, cl, alpha=0.25, s=12, color=PALETTE[2], zorder=3)
    ax.axhline(AMYLOID_POS_THRESHOLD, color="black", lw=1.5, ls=":",
               label=f"Threshold={AMYLOID_POS_THRESHOLD}")
    ax.set_xticks([1])
    ax.set_xticklabels(["All Subjects"])
    ax.set_ylabel("Centiloid Score (CL)")
    ax.set_title("Boxplot with Jittered Points")
    ax.legend(fontsize=10)

    # Annotation text box
    stats_text = (
        f"N={len(cl)}  |  Mean={cl.mean():.1f}  |  Std={cl.std():.1f}\n"
        f"Skew={cl.skew():.2f}  |  Kurt={cl.kurtosis():.2f}\n"
        f"Range=[{cl.min():.1f}, {cl.max():.1f}]"
    )
    fig.text(0.5, 0.01, stats_text, ha="center", fontsize=10,
             bbox=dict(facecolor="lightyellow", edgecolor="gray", alpha=0.8))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    path = os.path.join(out_dir, "01_overall_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


def plot_train_val_comparison(train: pd.DataFrame, val: pd.DataFrame,
                               out_dir: str):
    """Side-by-side comparison of train vs validation distributions."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Train vs Validation — Centiloid Distribution",
                 fontsize=15, fontweight="bold")

    # --- Overlay KDEs ---
    ax = axes[0]
    for df, label, color in [(train, f"Train (n={len(train)})", PALETTE[0]),
                              (val,   f"Val   (n={len(val)})",  PALETTE[1])]:
        cl = df["CENTILOIDS"]
        ax.hist(cl, bins=35, density=True, alpha=0.45,
                color=color, edgecolor="white")
        kde_x = np.linspace(cl.min()-5, cl.max()+5, 300)
        ax.plot(kde_x, stats.gaussian_kde(cl)(kde_x),
                color=color, lw=2.5, label=label)
    ax.axvline(AMYLOID_POS_THRESHOLD, color="black", lw=1.5,
               ls=":", label="Threshold")
    ax.set_xlabel("Centiloid Score (CL)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution Overlap")
    ax.legend()

    # --- Boxplots side-by-side ---
    ax = axes[1]
    combined_plot = pd.concat([train, val])
    combined_plot["split_label"] = combined_plot["split"].map(
        {"train": f"Train\n(n={len(train)})", "val": f"Val\n(n={len(val)})"})
    bp = combined_plot.boxplot(column="CENTILOIDS", by="split_label",
                                ax=ax, patch_artist=True,
                                return_type="dict")
    for patch, color in zip(bp["CENTILOIDS"]["boxes"],
                             [PALETTE[0], PALETTE[1]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.axhline(AMYLOID_POS_THRESHOLD, color="black", lw=1.5, ls=":")
    ax.set_title("Boxplot Comparison")
    ax.set_xlabel("")
    ax.set_ylabel("Centiloid Score (CL)")
    plt.sca(ax)
    plt.title("Boxplot Comparison")

    # --- Q-Q Plot (train quantiles vs val quantiles) ---
    ax = axes[2]
    t_sorted = np.sort(train["CENTILOIDS"].values)
    v_sorted  = np.sort(val["CENTILOIDS"].values)
    # Interpolate to same length
    t_interp = np.interp(
        np.linspace(0, 1, 100),
        np.linspace(0, 1, len(t_sorted)), t_sorted)
    v_interp = np.interp(
        np.linspace(0, 1, 100),
        np.linspace(0, 1, len(v_sorted)), v_sorted)
    ax.scatter(t_interp, v_interp, alpha=0.7, color=PALETTE[2], s=30)
    lim = [min(t_interp.min(), v_interp.min()) - 5,
           max(t_interp.max(), v_interp.max()) + 5]
    ax.plot(lim, lim, "r--", lw=1.5, label="y = x (perfect match)")
    ax.set_xlabel("Train Quantiles (CL)")
    ax.set_ylabel("Val Quantiles (CL)")
    ax.set_title("Q-Q Plot: Train vs Val")
    ax.legend()

    # KS test
    ks_stat, ks_p = stats.ks_2samp(
        train["CENTILOIDS"], val["CENTILOIDS"])
    fig.text(0.5, 0.01,
             f"KS Test: statistic={ks_stat:.4f}, p-value={ks_p:.4f} "
             f"({'distributions match' if ks_p > 0.05 else 'distributions differ'})",
             ha="center", fontsize=11,
             bbox=dict(facecolor="lightyellow", edgecolor="gray", alpha=0.8))

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    path = os.path.join(out_dir, "02_train_val_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


def plot_amyloid_pos_neg(combined: pd.DataFrame, out_dir: str):
    """Amyloid positive / negative split analysis."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Amyloid Positive vs Negative (Threshold = {AMYLOID_POS_THRESHOLD} CL)",
                 fontsize=15, fontweight="bold")

    pos = combined[combined["CENTILOIDS"] >= AMYLOID_POS_THRESHOLD]
    neg = combined[combined["CENTILOIDS"] <  AMYLOID_POS_THRESHOLD]

    # --- Pie chart ---
    ax = axes[0]
    sizes  = [len(pos), len(neg)]
    labels = [f"Positive\n(n={len(pos)}, {100*len(pos)/len(combined):.1f}%)",
              f"Negative\n(n={len(neg)}, {100*len(neg)/len(combined):.1f}%)"]
    ax.pie(sizes, labels=labels,
           colors=[PALETTE[3], PALETTE[4]],
           autopct="%1.1f%%", startangle=90,
           wedgeprops=dict(edgecolor="white", linewidth=2))
    ax.set_title("Overall Class Balance")

    # --- Split-level bar ---
    ax = axes[1]
    data = combined.groupby(["split", "amyloid_status"]).size().unstack(fill_value=0)
    data_pct = data.div(data.sum(axis=1), axis=0) * 100
    data_pct.plot(kind="bar", ax=ax,
                  color=[PALETTE[3], PALETTE[4]],
                  edgecolor="white", alpha=0.85)
    ax.set_title("Amyloid Status per Split (%)")
    ax.set_xlabel("")
    ax.set_ylabel("Percentage (%)")
    ax.set_xticklabels(["Train", "Val"], rotation=0)
    ax.legend(title="Status", fontsize=10)

    # --- Histogram colored by status ---
    ax = axes[2]
    ax.hist(neg["CENTILOIDS"], bins=30, density=True, alpha=0.6,
            color=PALETTE[4], edgecolor="white", label="Negative")
    ax.hist(pos["CENTILOIDS"], bins=30, density=True, alpha=0.6,
            color=PALETTE[3], edgecolor="white", label="Positive")
    ax.axvline(AMYLOID_POS_THRESHOLD, color="black",
               lw=2, ls="--", label="Threshold")
    ax.set_xlabel("Centiloid Score (CL)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution by Amyloid Status")
    ax.legend()

    plt.tight_layout()
    path = os.path.join(out_dir, "03_amyloid_pos_neg.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


def plot_cumulative_distribution(combined: pd.DataFrame, out_dir: str):
    """CDF and percentile rank analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Cumulative Distribution & Percentile Analysis",
                 fontsize=15, fontweight="bold")

    cl_sorted = np.sort(combined["CENTILOIDS"].values)
    cdf       = np.arange(1, len(cl_sorted)+1) / len(cl_sorted)

    ax = axes[0]
    ax.plot(cl_sorted, cdf * 100, color=PALETTE[0], lw=2.5)
    ax.axvline(AMYLOID_POS_THRESHOLD, color="red", lw=1.8, ls="--",
               label=f"Threshold={AMYLOID_POS_THRESHOLD} CL")
    # Mark percentiles
    for p, ls in [(25, ":"), (50, "-."), (75, "--")]:
        pval = np.percentile(cl_sorted, p)
        ax.axvline(pval, color="gray", lw=1.2, ls=ls)
        ax.axhline(p,    color="gray", lw=1.2, ls=ls)
        ax.annotate(f"P{p}={pval:.1f}", xy=(pval, p),
                    xytext=(pval+3, p+2), fontsize=9, color="gray")
    ax.set_xlabel("Centiloid Score (CL)")
    ax.set_ylabel("Cumulative % of Subjects")
    ax.set_title("Empirical CDF")
    ax.legend()
    ax.set_ylim(0, 102)

    # Percentile rank table
    ax = axes[1]
    ax.axis("off")
    pcts = [5, 10, 15, 20, 25, 50, 75, 80, 85, 90, 95, 99]
    rows = [[f"P{p}", f"{np.percentile(combined['CENTILOIDS'], p):.2f} CL"]
            for p in pcts]
    table = ax.table(
        cellText=rows,
        colLabels=["Percentile", "Centiloid Value"],
        loc="center", cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.5, 1.8)
    ax.set_title("Percentile Reference Table", fontsize=13, pad=20)

    plt.tight_layout()
    path = os.path.join(out_dir, "04_cumulative_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


def save_summary_stats(train, val, combined, out_dir):
    lines = []
    for df, label in [(combined, "ALL (Train+Val)"),
                       (train,    "TRAIN"),
                       (val,      "VAL")]:
        cl = df["CENTILOIDS"]
        pos_n = (cl >= AMYLOID_POS_THRESHOLD).sum()
        lines += [
            f"\n{'='*50}",
            f"  {label}  (n={len(df)})",
            f"{'='*50}",
            f"  Mean        : {cl.mean():.4f} CL",
            f"  Median      : {cl.median():.4f} CL",
            f"  Std         : {cl.std():.4f} CL",
            f"  Min         : {cl.min():.4f} CL",
            f"  Max         : {cl.max():.4f} CL",
            f"  Skewness    : {cl.skew():.4f}",
            f"  Kurtosis    : {cl.kurtosis():.4f}",
            f"  Amyloid Pos : {pos_n} ({100*pos_n/len(df):.2f}%)",
            f"  Amyloid Neg : {len(df)-pos_n} ({100*(len(df)-pos_n)/len(df):.2f}%)",
        ]
    ks_stat, ks_p = stats.ks_2samp(train["CENTILOIDS"], val["CENTILOIDS"])
    lines += [
        f"\n{'='*50}",
        "  KS Test: Train vs Val",
        f"{'='*50}",
        f"  KS Statistic : {ks_stat:.6f}",
        f"  p-value      : {ks_p:.6f}",
        f"  Conclusion   : {'Distributions match (p>0.05)' if ks_p>0.05 else 'Distributions differ (p≤0.05)'}",
    ]
    path = os.path.join(out_dir, "summary_stats.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="EDA 01 — Centiloid Distribution Analysis")
    parser.add_argument("--train_csv", default="data/train.csv")
    parser.add_argument("--val_csv",   default="data/val.csv")
    parser.add_argument("--out_dir",   default="results/eda/pre_train/01_centiloid_distribution")
    parser.add_argument("--tag",       default=None,
                        help="Label stamped on every figure "
                             "(e.g. run/experiment name). "
                             "Defaults to basename(out_dir).")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_run_tag(args.tag or os.path.basename(args.out_dir.rstrip("/\\")))
    print(f"\n{'#'*55}")
    print(f"  EDA 01 — Centiloid Distribution Analysis")
    print(f"  Output → {args.out_dir}")
    print(f"{'#'*55}")

    train, val, combined = load_data(args.train_csv, args.val_csv)

    # Console summaries
    print_summary(combined, "ALL (Train + Val)")
    print_summary(train,    "TRAIN")
    print_summary(val,      "VAL")

    # Plots
    print("\n[+] Generating plots...")
    plot_overall_distribution(combined, args.out_dir)
    plot_train_val_comparison(train, val, args.out_dir)
    plot_amyloid_pos_neg(combined, args.out_dir)
    plot_cumulative_distribution(combined, args.out_dir)
    save_summary_stats(train, val, combined, args.out_dir)

    print(f"\n✓ All outputs saved to → {args.out_dir}\n")


if __name__ == "__main__":
    main()