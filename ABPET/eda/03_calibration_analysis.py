"""
03_calibration_analysis.py
==========================
EDA Script 3 — Volume Intensity & Calibration Analysis

Justifies : TracerNorm magnitude (per-tracer μ shift) [model.py],
            foreground-aware preprocessing assumptions [README §preprocessing]
Reads     : train.csv, val.csv  (npy_path, TRACER.AMY, CENTILOIDS)
Optional  : --cohort_csv mapping ID → {NACC, A4} for cohort plots
Writes    : results/eda/03_calibration_analysis/

Examines voxel-level intensity characteristics of PET volumes:
  - Voxel intensity distributions per tracer
  - Brain foreground fraction (non-background voxel ratio)
  - NACC vs A4 cohort comparison  (only if cohort metadata is provided)
  - Centiloid vs mean / max intensity correlation (sanity check)

Outputs:
    results/eda/03_calibration_analysis/
        ├── 01_voxel_intensity_per_tracer.png
        ├── 02_foreground_fraction.png
        ├── 03_cohort_comparison.png        ← skipped if no cohort metadata
        ├── 04_intensity_centiloid_correlation.png
        └── calibration_report.txt

NOTE: Samples N_SAMPLES volumes per tracer to keep runtime manageable.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm

# Shared EDA constants & style
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (
    AMYLOID_POS_THRESHOLD, TRACER_ORDER, TRACER_COLORS,
    FOREGROUND_THRESH, setup_style, set_run_tag, log,
)

setup_style()


def load_data(train_csv, val_csv):
    train = pd.read_csv(train_csv); train["split"] = "train"
    val   = pd.read_csv(val_csv);   val["split"]   = "val"
    combined = pd.concat([train, val], ignore_index=True)
    return combined


def load_cohort(df: pd.DataFrame, cohort_csv: str | None) -> tuple[pd.DataFrame, bool]:
    """
    Attach a `cohort` column to `df` from real metadata (no fabrication).

    Resolution order:
      1. If `cohort_csv` is provided and has columns {ID, cohort} → join on ID.
      2. Else if npy_path contains 'NACC' or 'A4' substrings → infer from path.
      3. Otherwise → return df unchanged and (has_cohort=False).

    Returns:
        (df, has_cohort) — `has_cohort` is False when no source was available;
        callers must skip cohort-stratified plots in that case.
    """
    df = df.copy()

    if cohort_csv and os.path.exists(cohort_csv):
        cmap = pd.read_csv(cohort_csv)
        assert {"ID", "cohort"} <= set(cmap.columns), \
            f"cohort_csv must have ID, cohort columns. Got: {list(cmap.columns)}"
        df = df.merge(cmap[["ID", "cohort"]], on="ID", how="left")
        log(f"Loaded cohort metadata from {cohort_csv}", "ok")
        return df, True

    if df["npy_path"].str.contains("NACC|A4", na=False).any():
        df["cohort"] = df["npy_path"].apply(
            lambda p: "NACC" if "NACC" in str(p) else "A4"
        )
        log("Inferred cohort from npy_path substrings (NACC / A4)", "ok")
        return df, True

    log("No cohort metadata available — cohort plots will be skipped", "warn")
    return df, False


def extract_volume_stats(df: pd.DataFrame, n_samples: int) -> pd.DataFrame:
    """Sample up to n_samples volumes per tracer and compute voxel statistics."""
    records = []
    # Concatenate per-tracer samples (avoids pandas groupby.apply column-drop quirks)
    sample_df = pd.concat([
        g.sample(min(n_samples, len(g)), random_state=42)
        for _, g in df.groupby("TRACER.AMY")
    ], ignore_index=True)
    print(f"  Loading {len(sample_df)} volumes...")
    for _, row in tqdm(sample_df.iterrows(), total=len(sample_df)):
        try:
            vol = np.load(row["npy_path"])[0]   # (128,128,128)
            fg  = vol[vol > FOREGROUND_THRESH]   # foreground mask
            rec = {
                "ID"           : row["ID"],
                "TRACER.AMY"   : row["TRACER.AMY"],
                "CENTILOIDS"   : row["CENTILOIDS"],
                "split"        : row["split"],
                "vol_mean"     : float(vol.mean()),
                "vol_std"      : float(vol.std()),
                "vol_max"      : float(vol.max()),
                "vol_p95"      : float(np.percentile(vol, 95)),
                "vol_p99"      : float(np.percentile(vol, 99)),
                "fg_fraction"  : float((vol > FOREGROUND_THRESH).mean()),
                "fg_mean"      : float(fg.mean()) if len(fg) > 0 else np.nan,
                "fg_std"       : float(fg.std())  if len(fg) > 0 else np.nan,
            }
            if "cohort" in row.index:
                rec["cohort"] = row["cohort"]
            records.append(rec)
        except Exception as e:
            log(f"Could not load {row['npy_path']}: {e}", "warn")

    return pd.DataFrame(records)


def plot_voxel_intensity_per_tracer(stats_df: pd.DataFrame, out_dir: str):
    """Violin plots of voxel-level summary statistics per tracer."""
    metrics = {
        "vol_mean":    "Volume Mean Intensity",
        "vol_std":     "Volume Std Intensity",
        "vol_p95":     "Volume P95 Intensity",
        "fg_mean":     "Foreground Mean Intensity",
    }
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Voxel Intensity Statistics per Tracer",
                 fontsize=16, fontweight="bold")
    axes = axes.flatten()

    palette = {t: TRACER_COLORS[t] for t in TRACER_ORDER}
    for i, (metric, title) in enumerate(metrics.items()):
        ax = axes[i]
        sns.violinplot(data=stats_df, x="TRACER.AMY", y=metric,
                       hue="TRACER.AMY", hue_order=TRACER_ORDER,
                       order=TRACER_ORDER, palette=palette,
                       ax=ax, inner="box", cut=0, alpha=0.8, legend=False)
        sns.stripplot(data=stats_df, x="TRACER.AMY", y=metric,
                      order=TRACER_ORDER, color="black",
                      ax=ax, alpha=0.2, size=3, jitter=True)
        ax.set_title(title)
        ax.set_xlabel("Radiotracer")
        ax.set_ylabel("Intensity (normalized [0,1])")

        # Annotate means
        for j, tracer in enumerate(TRACER_ORDER):
            m = stats_df[stats_df["TRACER.AMY"] == tracer][metric].mean()
            ax.text(j, stats_df[metric].max() * 1.02,
                    f"μ={m:.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    path = os.path.join(out_dir, "01_voxel_intensity_per_tracer.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


def plot_foreground_fraction(stats_df: pd.DataFrame, out_dir: str):
    """Brain coverage (foreground fraction) analysis."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Brain Foreground Fraction Analysis "
                 f"(Threshold = {FOREGROUND_THRESH})",
                 fontsize=15, fontweight="bold")

    palette = {t: TRACER_COLORS[t] for t in TRACER_ORDER}

    # Histogram per tracer
    ax = axes[0]
    for tracer in TRACER_ORDER:
        vals = stats_df[stats_df["TRACER.AMY"] == tracer]["fg_fraction"]
        ax.hist(vals, bins=20, density=True, alpha=0.55,
                color=TRACER_COLORS[tracer], edgecolor="white",
                label=f"{tracer} (μ={vals.mean():.3f})")
    ax.set_xlabel("Foreground Fraction")
    ax.set_ylabel("Density")
    ax.set_title("Foreground Fraction Distribution")
    ax.legend(fontsize=10)

    # Boxplot
    ax = axes[1]
    sns.boxplot(data=stats_df, x="TRACER.AMY", y="fg_fraction",
                hue="TRACER.AMY", hue_order=TRACER_ORDER,
                order=TRACER_ORDER, palette=palette, ax=ax,
                width=0.5, fliersize=4, legend=False)
    ax.set_title("Foreground Fraction per Tracer")
    ax.set_xlabel("Radiotracer")
    ax.set_ylabel("Fraction of Voxels > threshold")

    # Foreground fraction vs centiloid
    ax = axes[2]
    for tracer in TRACER_ORDER:
        sub = stats_df[stats_df["TRACER.AMY"] == tracer]
        ax.scatter(sub["fg_fraction"], sub["CENTILOIDS"],
                   color=TRACER_COLORS[tracer], alpha=0.5,
                   s=20, label=tracer)
    ax.set_xlabel("Foreground Fraction")
    ax.set_ylabel("Centiloid Score")
    ax.set_title("Foreground Fraction vs Centiloid")
    ax.legend(fontsize=10)

    plt.tight_layout()
    path = os.path.join(out_dir, "02_foreground_fraction.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


def plot_cohort_comparison(df: pd.DataFrame,
                            stats_df: pd.DataFrame,
                            out_dir: str):
    """NACC vs A4 cohort: centiloid + intensity comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Cohort Comparison: NACC vs A4",
                 fontsize=15, fontweight="bold")

    cohort_pal = {"NACC": "#5B9BD5", "A4": "#ED7D31"}

    # Centiloid distribution
    ax = axes[0]
    for cohort, color in cohort_pal.items():
        vals = df[df["cohort"] == cohort]["CENTILOIDS"]
        ax.hist(vals, bins=30, density=True, alpha=0.55,
                color=color, edgecolor="white", label=f"{cohort} (n={len(vals)})")
        if len(vals) > 1:
            kde_x = np.linspace(vals.min()-5, vals.max()+5, 300)
            ax.plot(kde_x, stats.gaussian_kde(vals)(kde_x), color=color, lw=2)
    ax.set_title("Centiloid Distribution by Cohort")
    ax.set_xlabel("Centiloid Score (CL)")
    ax.set_ylabel("Density")
    ax.legend()

    # Volume mean intensity by cohort
    ax = axes[1]
    sns.boxplot(data=stats_df, x="cohort", y="vol_mean",
                hue="cohort", palette=cohort_pal, ax=ax,
                width=0.4, fliersize=4, legend=False)
    sns.stripplot(data=stats_df, x="cohort", y="vol_mean",
                  color="black", ax=ax, alpha=0.2, size=3)
    ax.set_title("Volume Mean Intensity by Cohort")
    ax.set_xlabel("Cohort")
    ax.set_ylabel("Mean Voxel Intensity")

    # Tracer composition per cohort
    ax = axes[2]
    comp = df.groupby(["cohort", "TRACER.AMY"]).size().unstack(fill_value=0)
    comp_pct = comp.div(comp.sum(axis=1), axis=0) * 100
    comp_pct.T.plot(kind="bar", ax=ax,
                    color=[cohort_pal["NACC"], cohort_pal["A4"]],
                    edgecolor="white", alpha=0.85)
    ax.set_title("Tracer Composition per Cohort (%)")
    ax.set_xlabel("Radiotracer")
    ax.set_ylabel("Percentage (%)")
    ax.set_xticklabels(TRACER_ORDER, rotation=0)
    ax.legend(title="Cohort")

    plt.tight_layout()
    path = os.path.join(out_dir, "03_cohort_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


def plot_intensity_centiloid_correlation(stats_df: pd.DataFrame, out_dir: str):
    """Correlation between volume intensity stats and centiloid target."""
    intensity_metrics = ["vol_mean", "vol_std", "vol_p95",
                          "vol_p99", "fg_mean", "fg_fraction"]
    palette = {t: TRACER_COLORS[t] for t in TRACER_ORDER}

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Intensity Statistics vs Centiloid Score\n"
                 "(per-tracer Pearson r annotated)",
                 fontsize=15, fontweight="bold")
    axes = axes.flatten()

    for i, metric in enumerate(intensity_metrics):
        ax = axes[i]
        for tracer in TRACER_ORDER:
            sub = stats_df[stats_df["TRACER.AMY"] == tracer].dropna(
                subset=[metric])
            if len(sub) < 3:
                continue
            ax.scatter(sub[metric], sub["CENTILOIDS"],
                       color=TRACER_COLORS[tracer], alpha=0.4, s=20)
            r, p = stats.pearsonr(sub[metric], sub["CENTILOIDS"])
            # Regression line
            m, b  = np.polyfit(sub[metric], sub["CENTILOIDS"], 1)
            x_lin = np.linspace(sub[metric].min(), sub[metric].max(), 100)
            ax.plot(x_lin, m * x_lin + b,
                    color=TRACER_COLORS[tracer], lw=1.5, alpha=0.8,
                    label=f"{tracer} r={r:.2f}")

        # Overall correlation
        clean = stats_df.dropna(subset=[metric])
        if len(clean) > 2:
            r_all, _ = stats.pearsonr(clean[metric], clean["CENTILOIDS"])
            ax.text(0.03, 0.97, f"Overall r={r_all:.3f}",
                    transform=ax.transAxes, va="top", fontsize=10,
                    bbox=dict(fc="white", ec="gray", alpha=0.7))

        ax.set_xlabel(metric.replace("_", " ").title())
        ax.set_ylabel("Centiloid Score (CL)")
        ax.set_title(metric.replace("_", " ").title())
        ax.legend(fontsize=8, ncol=2)

    plt.tight_layout()
    path = os.path.join(out_dir, "04_intensity_centiloid_correlation.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


def save_calibration_report(df, stats_df, out_dir):
    lines = [
        "Calibration Analysis Report",
        "=" * 65,
        f"Total subjects sampled : {len(stats_df)}",
        "",
        f"{'Tracer':<6} {'N':>4} | {'Mean μ':>8} {'Std σ':>8} "
        f"{'P95':>8} {'FG_frac':>9} {'FG_mean':>9}",
        "-" * 65,
    ]
    for tracer in TRACER_ORDER:
        s = stats_df[stats_df["TRACER.AMY"] == tracer]
        lines.append(
            f"{tracer:<6} {len(s):>4} | "
            f"{s['vol_mean'].mean():>8.4f} "
            f"{s['vol_std'].mean():>8.4f} "
            f"{s['vol_p95'].mean():>8.4f} "
            f"{s['fg_fraction'].mean():>9.4f} "
            f"{s['fg_mean'].mean():>9.4f}"
        )
    lines += ["=" * 65, ""]

    # ANOVA test across tracers for vol_mean
    groups   = [stats_df[stats_df["TRACER.AMY"] == t]["vol_mean"].dropna()
                for t in TRACER_ORDER if len(stats_df[stats_df["TRACER.AMY"] == t]) > 1]
    f, p_anova = stats.f_oneway(*groups)
    lines += [
        "One-Way ANOVA (vol_mean across tracers):",
        f"  F-statistic = {f:.4f}",
        f"  p-value     = {p_anova:.6f}",
        f"  Result: {'Significant tracer effect (p<0.05)' if p_anova<0.05 else 'No significant tracer effect'}",
    ]

    path = os.path.join(out_dir, "calibration_report.txt")
    with open(path, "w") as f_out:
        f_out.write("\n".join(lines))
    print(f"  Saved → {path}")
    print("\n" + "\n".join(lines))


def main():
    parser = argparse.ArgumentParser(
        description="EDA 03 — Volume Intensity & Calibration Analysis")
    parser.add_argument("--train_csv",  default="data/train.csv")
    parser.add_argument("--val_csv",    default="data/val.csv")
    parser.add_argument("--out_dir",    default="results/eda/pre_train/03_calibration_analysis")
    parser.add_argument("--n_samples",  type=int, default=50,
                        help="Volumes to sample per tracer (default=50)")
    parser.add_argument("--cohort_csv", default=None,
                        help="Optional CSV mapping ID → cohort (NACC/A4). "
                             "If omitted, cohort plots are skipped.")
    parser.add_argument("--tag",        default=None,
                        help="Label stamped on every figure "
                             "(e.g. run/experiment name). "
                             "Defaults to basename(out_dir).")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_run_tag(args.tag or os.path.basename(args.out_dir.rstrip("/\\")))
    print(f"\n{'#'*55}")
    print("  EDA 03 — Calibration Analysis")
    print(f"  Sampling {args.n_samples} volumes per tracer")
    print(f"  Output → {args.out_dir}")
    print(f"{'#'*55}")

    df = load_data(args.train_csv, args.val_csv)
    df, has_cohort = load_cohort(df, args.cohort_csv)
    stats_df = extract_volume_stats(df, args.n_samples)

    if has_cohort:
        stats_df = stats_df.merge(
            df[["ID", "cohort"]].drop_duplicates(), on="ID", how="left",
            suffixes=("", "_dup"),
        )
        if "cohort_dup" in stats_df.columns:
            stats_df["cohort"] = stats_df["cohort"].fillna(stats_df["cohort_dup"])
            stats_df.drop("cohort_dup", axis=1, inplace=True)

    print("\n[+] Generating plots...")
    plot_voxel_intensity_per_tracer(stats_df, args.out_dir)
    plot_foreground_fraction(stats_df, args.out_dir)
    if has_cohort:
        plot_cohort_comparison(df, stats_df, args.out_dir)
    else:
        log("Skipping cohort comparison plot (no cohort metadata)", "skip")
    plot_intensity_centiloid_correlation(stats_df, args.out_dir)
    save_calibration_report(df, stats_df, args.out_dir)

    print(f"\n✓ All outputs saved to → {args.out_dir}\n")


if __name__ == "__main__":
    main()