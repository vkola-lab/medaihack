"""
02_tracer_comparison.py
=======================
EDA Script 2 — Radiotracer Comparison Analysis

Justifies : TracerNorm + FiLM conditioning [model.py],
            per-tracer augmentation strength [train.py / augmentation.py]
Reads     : train.csv, val.csv  (TRACER.AMY, CENTILOIDS, npy_path)
Writes    : results/eda/02_tracer_comparison/

Compares centiloid distributions, sample counts, and pairwise statistical
distance across the four radiotracers (FBP, FBB, NAV, PIB), and samples
representative PET slices per tracer for qualitative intensity comparison.

Outputs:
    results/eda/02_tracer_comparison/
        ├── 01_tracer_centiloid_distribution.png
        ├── 02_tracer_sample_counts.png
        ├── 03_tracer_violin_boxplot.png
        ├── 04_tracer_pairwise_stats.png
        ├── 05_tracer_pet_slices.png     ← visual sanity check
        └── tracer_summary.txt
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
from itertools import combinations

# Shared EDA constants & style
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (
    AMYLOID_POS_THRESHOLD, TRACER_ORDER, TRACER_NAMES, TRACER_COLORS,
    setup_style, set_run_tag, log,
)

setup_style()


def load_data(train_csv, val_csv):
    train = pd.read_csv(train_csv); train["split"] = "train"
    val   = pd.read_csv(val_csv);   val["split"]   = "val"
    combined = pd.concat([train, val], ignore_index=True)
    combined["amyloid_pos"] = combined["CENTILOIDS"] >= AMYLOID_POS_THRESHOLD
    return combined


def plot_tracer_centiloid_distribution(df: pd.DataFrame, out_dir: str):
    """KDE overlay + histogram per tracer."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Centiloid Distribution by Radiotracer",
                 fontsize=16, fontweight="bold")
    axes = axes.flatten()

    for i, tracer in enumerate(TRACER_ORDER):
        ax    = axes[i]
        sub   = df[df["TRACER.AMY"] == tracer]["CENTILOIDS"]
        color = TRACER_COLORS[tracer]

        ax.hist(sub, bins=25, density=True, alpha=0.55,
                color=color, edgecolor="white")
        kde_x = np.linspace(sub.min()-5, sub.max()+5, 300)
        ax.plot(kde_x, stats.gaussian_kde(sub)(kde_x),
                color=color, lw=2.5)
        ax.axvline(sub.mean(),   color="red",    lw=1.8, ls="--",
                   label=f"Mean={sub.mean():.1f}")
        ax.axvline(sub.median(), color="orange", lw=1.8, ls="-.",
                   label=f"Med={sub.median():.1f}")
        ax.axvline(AMYLOID_POS_THRESHOLD, color="black", lw=1.3,
                   ls=":", label="Threshold")

        # Amyloid pos % annotation
        pos_pct = 100 * (sub >= AMYLOID_POS_THRESHOLD).mean()
        ax.text(0.98, 0.97,
                f"n={len(sub)}\n+Amyloid: {pos_pct:.1f}%\n"
                f"Skew: {sub.skew():.2f}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=10, bbox=dict(fc="white", ec="gray", alpha=0.7))

        ax.set_title(f"{tracer} — {TRACER_NAMES[tracer]}", fontsize=12)
        ax.set_xlabel("Centiloid Score (CL)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(out_dir, "01_tracer_centiloid_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


def plot_tracer_sample_counts(df: pd.DataFrame, out_dir: str):
    """Bar charts showing sample counts per tracer per split."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Sample Counts by Tracer and Split",
                 fontsize=15, fontweight="bold")

    counts = df.groupby(["TRACER.AMY", "split"]).size().unstack(fill_value=0)
    counts = counts.reindex(TRACER_ORDER)

    # --- Total counts ---
    ax = axes[0]
    total = df.groupby("TRACER.AMY").size().reindex(TRACER_ORDER)
    bars  = ax.bar(TRACER_ORDER, total.values,
                   color=[TRACER_COLORS[t] for t in TRACER_ORDER],
                   edgecolor="white", alpha=0.85)
    for bar, v in zip(bars, total.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8,
                str(v), ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_title("Total Samples per Tracer")
    ax.set_ylabel("Count")
    ax.set_xlabel("Radiotracer")

    # --- Train / Val stacked ---
    ax = axes[1]
    bottom = np.zeros(len(TRACER_ORDER))
    for split, color in [("train", "#5B9BD5"), ("val", "#ED7D31")]:
        vals = counts.get(split, pd.Series(0, index=TRACER_ORDER)).values
        bars = ax.bar(TRACER_ORDER, vals, bottom=bottom,
                      label=split.capitalize(), color=color,
                      alpha=0.85, edgecolor="white")
        bottom += vals
    ax.set_title("Train vs Val per Tracer")
    ax.set_ylabel("Count")
    ax.set_xlabel("Radiotracer")
    ax.legend()

    # --- Amyloid pos / neg stacked ---
    ax = axes[2]
    pos_counts = df[df["amyloid_pos"]].groupby(
        "TRACER.AMY").size().reindex(TRACER_ORDER, fill_value=0)
    neg_counts = df[~df["amyloid_pos"]].groupby(
        "TRACER.AMY").size().reindex(TRACER_ORDER, fill_value=0)
    ax.bar(TRACER_ORDER, neg_counts.values,
           label="Amyloid Negative", color="#4CAF50", alpha=0.8, edgecolor="white")
    ax.bar(TRACER_ORDER, pos_counts.values, bottom=neg_counts.values,
           label="Amyloid Positive", color="#F44336", alpha=0.8, edgecolor="white")
    ax.set_title("Amyloid Status per Tracer")
    ax.set_ylabel("Count")
    ax.set_xlabel("Radiotracer")
    ax.legend()

    plt.tight_layout()
    path = os.path.join(out_dir, "02_tracer_sample_counts.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


def plot_violin_boxplot(df: pd.DataFrame, out_dir: str):
    """Violin + box + strip plot for centiloid per tracer."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Centiloid Distribution per Tracer — Violin & Boxplot",
                 fontsize=15, fontweight="bold")

    palette = {t: TRACER_COLORS[t] for t in TRACER_ORDER}

    # Violin
    ax = axes[0]
    sns.violinplot(data=df, x="TRACER.AMY", y="CENTILOIDS",
                   hue="TRACER.AMY", hue_order=TRACER_ORDER,
                   order=TRACER_ORDER, palette=palette, ax=ax,
                   inner="quartile", cut=0, alpha=0.8, legend=False)
    ax.axhline(AMYLOID_POS_THRESHOLD, color="black",
               lw=1.8, ls="--", label="Threshold")
    ax.set_title("Violin Plot")
    ax.set_xlabel("Radiotracer")
    ax.set_ylabel("Centiloid Score (CL)")
    ax.legend()

    # Box + strip
    ax = axes[1]
    sns.boxplot(data=df, x="TRACER.AMY", y="CENTILOIDS",
                hue="TRACER.AMY", hue_order=TRACER_ORDER,
                order=TRACER_ORDER, palette=palette, ax=ax,
                width=0.5, fliersize=0, legend=False)
    sns.stripplot(data=df, x="TRACER.AMY", y="CENTILOIDS",
                  order=TRACER_ORDER, color="black", ax=ax,
                  alpha=0.15, size=3, jitter=True)
    ax.axhline(AMYLOID_POS_THRESHOLD, color="red",
               lw=1.8, ls="--", label="Threshold")
    ax.set_title("Boxplot + Jittered Points")
    ax.set_xlabel("Radiotracer")
    ax.set_ylabel("Centiloid Score (CL)")
    ax.legend()

    # Add per-tracer n annotations
    for ax_obj in [axes[0], axes[1]]:
        for i, tracer in enumerate(TRACER_ORDER):
            n = (df["TRACER.AMY"] == tracer).sum()
            ax_obj.text(i, df["CENTILOIDS"].max() + 3,
                        f"n={n}", ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(out_dir, "03_tracer_violin_boxplot.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


def plot_pairwise_stats(df: pd.DataFrame, out_dir: str):
    """Pairwise statistical tests between tracers + correlation heatmap."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Pairwise Tracer Statistical Comparisons",
                 fontsize=15, fontweight="bold")

    pairs    = list(combinations(TRACER_ORDER, 2))
    ks_mat   = pd.DataFrame(np.nan, index=TRACER_ORDER, columns=TRACER_ORDER)
    mw_p_mat = pd.DataFrame(np.nan, index=TRACER_ORDER, columns=TRACER_ORDER)

    for t1, t2 in pairs:
        s1 = df[df["TRACER.AMY"] == t1]["CENTILOIDS"]
        s2 = df[df["TRACER.AMY"] == t2]["CENTILOIDS"]
        ks_stat, ks_p = stats.ks_2samp(s1, s2)
        mw_stat, mw_p = stats.mannwhitneyu(s1, s2, alternative="two-sided")
        ks_mat.loc[t1, t2]   = ks_stat
        ks_mat.loc[t2, t1]   = ks_stat
        mw_p_mat.loc[t1, t2] = mw_p
        mw_p_mat.loc[t2, t1] = mw_p

    # KS statistic heatmap
    ax = axes[0]
    mask = np.eye(len(TRACER_ORDER), dtype=bool)
    sns.heatmap(ks_mat.astype(float), annot=True, fmt=".3f",
                cmap="YlOrRd", ax=ax, mask=mask,
                linewidths=0.5, linecolor="white",
                cbar_kws={"label": "KS Statistic"})
    ax.set_title("KS Statistic\n(higher = more different distributions)")

    # Mann-Whitney p-value heatmap
    ax = axes[1]
    mw_log = -np.log10(mw_p_mat.astype(float) + 1e-300)
    sns.heatmap(mw_log, annot=mw_p_mat.round(4).astype(str),
                fmt="", cmap="Blues", ax=ax, mask=mask,
                linewidths=0.5, linecolor="white",
                cbar_kws={"label": "−log₁₀(p-value)"})
    ax.set_title("Mann-Whitney U Test\n−log₁₀(p) — annotated with p-value")

    plt.tight_layout()
    path = os.path.join(out_dir, "04_tracer_pairwise_stats.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


def plot_pet_slices(df: pd.DataFrame, out_dir: str, n_per_tracer: int = 3):
    """
    Load sample PET volumes and show axial, coronal, sagittal slices
    for each tracer side-by-side — qualitative intensity comparison.
    """
    print("  [+] Loading sample PET volumes for visualization...")

    fig = plt.figure(figsize=(4 * n_per_tracer, 4 * len(TRACER_ORDER)))
    fig.suptitle("Representative PET Slices by Tracer\n"
                 "(Axial mid-slice, min-max normalized)",
                 fontsize=15, fontweight="bold")

    gs = gridspec.GridSpec(len(TRACER_ORDER), n_per_tracer,
                           figure=fig, hspace=0.35, wspace=0.05)

    for row_i, tracer in enumerate(TRACER_ORDER):
        sub = df[df["TRACER.AMY"] == tracer].sample(
            min(n_per_tracer, len(df[df["TRACER.AMY"] == tracer])),
            random_state=42
        )
        for col_j, (_, record) in enumerate(sub.iterrows()):
            ax = fig.add_subplot(gs[row_i, col_j])
            try:
                vol = np.load(record["npy_path"])  # (1,128,128,128)
                vol = vol[0]                        # (128,128,128)
                mid = vol.shape[2] // 2             # axial mid-slice
                ax.imshow(vol[:, :, mid], cmap="hot",
                          vmin=0, vmax=1, origin="lower")
                cl_val = record["CENTILOIDS"]
                status = "+" if cl_val >= AMYLOID_POS_THRESHOLD else "−"
                ax.set_title(f"{tracer} [{status}]\nCL={cl_val:.1f}",
                              fontsize=9, pad=3)
            except Exception as e:
                ax.text(0.5, 0.5, f"Error\n{str(e)[:30]}",
                        ha="center", va="center", transform=ax.transAxes,
                        fontsize=8, color="red")
            ax.axis("off")

    path = os.path.join(out_dir, "05_tracer_pet_slices.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


def save_tracer_summary(df: pd.DataFrame, out_dir: str):
    lines = [
        "Radiotracer Summary Statistics",
        "=" * 60,
        f"{'Tracer':<8} {'N':>5} {'Mean':>8} {'Std':>8} {'Median':>8} "
        f"{'Min':>7} {'Max':>7} {'Skew':>7} {'Pos%':>6}",
        "-" * 60,
    ]
    for tracer in TRACER_ORDER:
        sub = df[df["TRACER.AMY"] == tracer]["CENTILOIDS"]
        pos_pct = 100 * (sub >= AMYLOID_POS_THRESHOLD).mean()
        lines.append(
            f"{tracer:<8} {len(sub):>5} {sub.mean():>8.2f} {sub.std():>8.2f} "
            f"{sub.median():>8.2f} {sub.min():>7.2f} {sub.max():>7.2f} "
            f"{sub.skew():>7.3f} {pos_pct:>5.1f}%"
        )
    lines.append("=" * 60)
    path = os.path.join(out_dir, "tracer_summary.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved → {path}")
    print("\n" + "\n".join(lines))


def main():
    parser = argparse.ArgumentParser(
        description="EDA 02 — Radiotracer Comparison Analysis")
    parser.add_argument("--train_csv", default="data/train.csv")
    parser.add_argument("--val_csv",   default="data/val.csv")
    parser.add_argument("--out_dir",   default="results/eda/pre_train/02_tracer_comparison")
    parser.add_argument("--n_slices",  type=int, default=3,
                        help="PET volumes per tracer to visualize")
    parser.add_argument("--tag",       default=None,
                        help="Label stamped on every figure "
                             "(e.g. run/experiment name). "
                             "Defaults to basename(out_dir).")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_run_tag(args.tag or os.path.basename(args.out_dir.rstrip("/\\")))
    print(f"\n{'#'*55}")
    print("  EDA 02 — Radiotracer Comparison Analysis")
    print(f"  Output → {args.out_dir}")
    print(f"{'#'*55}")

    df = load_data(args.train_csv, args.val_csv)

    print("\n[+] Generating plots...")
    plot_tracer_centiloid_distribution(df, args.out_dir)
    plot_tracer_sample_counts(df, args.out_dir)
    plot_violin_boxplot(df, args.out_dir)
    plot_pairwise_stats(df, args.out_dir)
    plot_pet_slices(df, args.out_dir, n_per_tracer=args.n_slices)
    save_tracer_summary(df, args.out_dir)

    print(f"\n✓ All outputs saved to → {args.out_dir}\n")


if __name__ == "__main__":
    main()