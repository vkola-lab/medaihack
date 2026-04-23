# MYGO · EDA Suite

***M**ultitracer-conditioned 3D resNet for am**Y**loid β-PET centil**O**id re**G**ression.*

Exploratory data analysis for **MYGO**, the team's submission to the
**MedAI Spring 2026 Amyloid β-PET Centiloid Prediction Challenge**
(Kolachalama Lab, Boston University). Each script in this directory
answers a single empirical question and produces the figures, tables,
and statistical tests that motivate the architectural choices in
`abpet/model/petresnet.py`, `abpet/nn/losses.py`, and `dev/train.py`.

The data are 2,000 training and 500 validation 3D PET volumes
(`(1, 128, 128, 128)`, float32, min-max normalised), each with a
ground-truth Centiloid score (Klunk et al., 2015) and a tracer label
drawn from {FBP, FBB, NAV, PIB}. Cohorts are **NACC** (Beekly et al.,2007) and **A4** (Sperling et al., 2014).

Challenge repository: <https://github.com/vkola-lab/medaihack>

---

## Contents

1. [Requirements](#1-requirements)
2. [Data](#2-data)
3. [Analyses](#3-analyses)
4. [Runbook](#4-runbook)
5. [Outputs](#5-outputs)
6. [From findings to Design](#6-findings--design)
7. [Reproducibility](#7-reproducibility)
8. [Limitations](#8-limitations)
9. [References](#9-references)

---

## 1. Requirements

| Package | Version | Purpose |
|---|---|---|
| `numpy` | ≥ 1.26 | arrays |
| `pandas` | ≥ 2.0 | dataframes |
| `matplotlib` | ≥ 3.7 | figures |
| `seaborn` | ≥ 0.13 | violin / box / heatmap |
| `scipy` | ≥ 1.10 | KS, Mann-Whitney U, ANOVA, Shapiro-Wilk, Pearson |
| `tqdm` | ≥ 4.65 | progress bar (script 03 only) |

`numpy`, `pandas`, and `matplotlib` ship with the project
`requirements.txt`. `scipy` is available transitively via
`scikit-learn`. Install the remainder with:

```bash
pip install seaborn tqdm
```

---

## 2. Data

### 2.1 Expected files

```
data/
├── train.csv        2,000 samples
└── val.csv            500 samples
```

### 2.2 Required columns

| Column | Type | Description |
|---|---|---|
| `ID` | str | unique sample identifier |
| `npy_path` | str | absolute path to a `(1, 128, 128, 128)` float32 `.npy` |
| `TRACER.AMY` | str | one of `FBP`, `FBB`, `NAV`, `PIB` |
| `CENTILOIDS` | float | regression target (typically −50 to 200+) |

### 2.3 Optional inputs

| Flag | Script | Columns | Behaviour if absent |
|---|---|---|---|
| `--cohort_csv` | `03` | `ID`, `cohort` ∈ {`NACC`, `A4`} | cohort plot is **skipped** (not fabricated) |
| `--pred_csv` | `04` | `ID`, `PREDICTED_CENTILOIDS` | falls back to a mean-predictor baseline |

---

## 3. Analyses

| # | Script | Question | Statistical methods | Justifies |
|---|---|---|---|---|
| 01 | `01_centiloid_distribution.py` | How is centiloid distributed across splits and amyloid-status groups? | mean, median, skew, excess kurtosis; KS 2-sample (train vs val); empirical CDF | `HuberLoss(δ=25)`, `WeightedRandomSampler` |
| 02 | `02_tracer_comparison.py` | Do the four tracers differ in centiloid and image intensity? | per-tracer KDE; pairwise KS (6 pairs); pairwise Mann-Whitney U; axial mid-slice samples | `TracerNorm`, `FiLM` conditioning, per-tracer augmentation |
| 03 | `03_calibration_analysis.py` | How do voxel intensities and brain coverage differ across tracers? | per-tracer voxel μ/σ/p95/p99; foreground fraction (>0.05); one-way ANOVA on `vol_mean` | `TracerNorm` magnitude; preprocessing assumptions |
| 04 | `04_model_error_analysis.py` | Where does the trained model fail? Per-tracer? Per CL bin? Versus naive baselines? | residual KDE + Q-Q + Shapiro-Wilk; Bland-Altman per tracer; stratified MAE; baseline comparison | post-training diagnostics → ablation feedback |

All scripts share the I/O contract:

- Accept `--train_csv`, `--val_csv`, `--out_dir` (script 04: `--val_csv` only).
- Create `out_dir` if missing.
- Write a `*.txt` report alongside every PNG.
- Exit 0 on success; non-zero on data-assertion or load failure.

The shared module **`_common.py`** is the single source of truth for:

| Symbol | Type | Value | Used by |
|---|---|---|---|
| `AMYLOID_POS_THRESHOLD` | `float` | `24.4` (Klunk 2015) | all |
| `FOREGROUND_THRESH` | `float` | `0.05` | 03 |
| `TRACER_ORDER` | `list[str]` | `["FBP", "FBB", "NAV", "PIB"]` | 02–04 |
| `TRACER_NAMES` | `dict` | tracer → full name | 02 |
| `TRACER_COLORS` | `dict` | tracer → seaborn `Set1` RGB | 02–04 |
| `setup_style(font_scale=1.2)` | `() → None` | seaborn `whitegrid` theme | all |
| `log(msg, level)` | callable | levels: `info`, `ok`, `warn`, `save`, `skip` | all |

---

## 4. Runbook

The scripts are independent and idempotent. They can be executed in any
order (script 04 needs predictions for the full path; otherwise it runs
the baseline path).

### 4.1 Set defaults

All scripts default to:

```text
--train_csv  data/train.csv
--val_csv    data/val.csv
--out_dir    results/eda/<script_name>/
```

Override per script with `--train_csv`, `--val_csv`, `--out_dir`.

### 4.2 Script 01 — Centiloid distribution

```bash
python eda/01_centiloid_distribution.py
```

### 4.3 Script 02 — Tracer comparison

```bash
python eda/02_tracer_comparison.py --n_slices 3
```

### 4.4 Script 03 — Calibration analysis

```bash
python eda/03_calibration_analysis.py \
    --n_samples 50                              # volumes per tracer
# Optional: provide cohort metadata
python eda/03_calibration_analysis.py \
    --cohort_csv data/cohort.csv
```

### 4.5 Script 04 — Model error analysis

```bash
# Without predictions — naive baseline
python eda/04_model_error_analysis.py

# With predictions
python eda/04_model_error_analysis.py \
    --pred_csv results/predictions.csv
```

### 4.6 Full sweep (sequential, fail-fast)

```bash
for s in eda/[0-9][0-9]_*.py; do python "$s" || exit 1; done
```

---

## 5. Outputs

```
results/eda/
├── 01_centiloid_distribution/
│   ├── 01_overall_distribution.png        histogram + KDE + boxplot
│   ├── 02_train_val_comparison.png        KDE overlay + Q-Q + KS test
│   ├── 03_amyloid_pos_neg.png             class balance per split
│   ├── 04_cumulative_distribution.png     empirical CDF + percentile table
│   └── summary_stats.txt                  per-split summary + KS report
│
├── 02_tracer_comparison/
│   ├── 01_tracer_centiloid_distribution.png
│   ├── 02_tracer_sample_counts.png
│   ├── 03_tracer_violin_boxplot.png
│   ├── 04_tracer_pairwise_stats.png       pairwise KS + Mann-Whitney heatmaps
│   ├── 05_tracer_pet_slices.png           axial mid-slice per tracer
│   └── tracer_summary.txt
│
├── 03_calibration_analysis/
│   ├── 01_voxel_intensity_per_tracer.png
│   ├── 02_foreground_fraction.png
│   ├── 03_cohort_comparison.png           (only if --cohort_csv)
│   ├── 04_intensity_centiloid_correlation.png
│   └── calibration_report.txt             per-tracer voxel stats + ANOVA
│
└── 04_model_error_analysis/
    ├── 01_predicted_vs_actual.png         scatter + identity line
    ├── 02_residual_distribution.png       6-panel residual diagnostics
    ├── 03_bland_altman.png                clinical agreement, per tracer
    ├── 04_error_by_centiloid_bin.png      stratified MAE
    ├── 05_baseline_comparison.png         model vs naive predictors
    └── error_report.txt
```

---

## 6. Findings → Design

The empirical findings produced by this suite are the sole motivation for
the architectural choices in the codebase. The trace from finding to
design choice is encoded in the `Justifies:` header of every script and
summarised here:

| Empirical finding | Source script | Design response | Implementation |
|---|---|---|---|
| Centiloid is right-skewed (median ≈ 10, IQR P25 ≈ −1.5 → P75 ≈ 47.2) | `01` | Robust regression with `δ ≈ IQR` | `HuberLoss(δ=25)` in `abpet/nn/losses.py` |
| 64.8 % of training samples are amyloid-negative (CL < 24.4) | `01` | Inverse-frequency oversampling | `WeightedRandomSampler` in `dev/train.py` |
| FBP voxel intensities are visibly brighter than FBB / NAV / PIB | `02`, `03` | Learned per-tracer (γ, β) intensity rescale | `TracerNorm` in `abpet/model/petresnet.py` |
| NAV is the most distributionally distant tracer (KS = 0.240 vs FBP) | `02` | Strong tracer conditioning at every backbone stage | `FiLMBlock` × 4 in `abpet/model/petresnet.py` |
| NAV has only n = 85 training samples | `02` | Stronger augmentation + higher head dropout | `build_train_transform(strong=True)` for NAV / PIB; `dropout_high=0.4` |
| Centiloid contains negative values (≥ −50) | `01` | No output activation | linear final FC in `abpet/model/petresnet.py` |

---

## 7. Reproducibility

| Source of randomness | Seed |
|---|---|
| `np.random.default_rng` (jitter, sampling) | `42` |
| `pandas.DataFrame.sample` | `42` |
| `stats.shapiro` subsample (script 04, `n ≤ 300`) | `42` |
| Matplotlib output | `dpi=150`, `bbox_inches="tight"` |

Every plot has a paired `*.txt` report so that the underlying numbers can
be audited without re-running the suite.

---

## 8. Limitations

- **Cohort metadata is optional.** Script 03 will *not* fabricate
  cohort labels if metadata is absent — the cohort comparison plot is
  skipped instead. Earlier versions of this suite inferred cohorts by
  row position; this practice has been removed.
- **No multiple-testing correction.** Script 02 reports six pairwise
  KS / Mann-Whitney p-values without Bonferroni or FDR adjustment.
  Effect sizes (KS statistic itself) are reported alongside p-values
  to support this trade-off.
- **Volume sampling is partial.** Script 03 samples 50 volumes per
  tracer by default to keep run time manageable; this is sufficient for
  intensity statistics but does not characterise rare per-volume
  artifacts.
- **One-way ANOVA assumes normality.** The `vol_mean` ANOVA in
  script 03's report should be cross-checked with a Kruskal-Wallis
  test for cohorts whose voxel-mean distribution is heavy-tailed.

---

## 9. References

**Methodology**

- Tukey JW. *Exploratory Data Analysis.* Addison-Wesley, 1977.
- Bland JM, Altman DG. Statistical methods for assessing agreement
  between two methods of clinical measurement. *The Lancet.*
  1986;327(8476):307–310.
- Huber PJ. Robust Estimation of a Location Parameter.
  *Annals of Mathematical Statistics.* 1964;35(1):73–101.

**Domain — amyloid PET and the Centiloid scale**

- Klunk WE, Koeppe RA, Price JC, et al. The Centiloid Project:
  standardizing quantitative amyloid plaque estimation by PET.
  *Alzheimer's & Dementia.* 2015;11(1):1–15.
- Beekly DL, Ramos EM, Lee WW, et al. The National Alzheimer's
  Coordinating Center (NACC) database: the Uniform Data Set.
  *Alzheimer Disease & Associated Disorders.* 2007;21(3):249–258.
- Sperling RA, Rentz DM, Johnson KA, et al. The A4 Study: Stopping AD
  Before Symptoms Begin? *Science Translational Medicine.*
  2014;6(228):228fs13.
- Jagust WJ, Landau SM, Koeppe RA, et al. The Alzheimer's Disease
  Neuroimaging Initiative 2 PET Core: 2015. *Alzheimer's & Dementia.*
  2015;11(7):757–771.

**Architecture inspirations** (cross-referenced from `abpet/model/petresnet.py`)

- Perez E, Strub F, de Vries H, Dumoulin V, Courville A. FiLM: Visual
  Reasoning with a General Conditioning Layer. *AAAI 2018.*
- He K, Zhang X, Ren S, Sun J. Deep Residual Learning for Image
  Recognition. *CVPR 2016.*
- Pérez-García F, Sparks R, Ourselin S. TorchIO: a Python library for
  efficient loading, preprocessing, augmentation and patch-based sampling
  of medical images in deep learning. *Computer Methods and Programs in
  Biomedicine.* 2021;208:106236.

**Challenge**

- Kolachalama Lab, Boston University. *MedAI Hackathon — Amyloid PET
  Centiloid Prediction.* Spring 2026.
  <https://github.com/vkola-lab/medaihack>
