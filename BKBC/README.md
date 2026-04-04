# Boston Kidney Biopsy Cohort — Hackathon Starter Code

## Task

Predict **acute tubular injury (ATI)** from plasma proteomics, as a binary classification problem:

| Label | Meaning |
|-------|---------|
| **0** | No ATI |
| **1** | ATI present |

Your model will be evaluated on an **external held-out test cohort** (KPMP) using **AUC** and **log loss**. Your goal is to build a model that generalises beyond the BKBC training data.

---

## Data

Training data is located at:

```
../../BKBC_train/train.csv
```

It contains **426 patients** (one row each) with the following columns:

| Column group | Description |
|---|---|
| `sample_id` | Anonymised patient identifier |
| `ati` | Binary ATI label (0 = No ATI, 1 = ATI) — your prediction target |
| `age` | Age (10-year bin midpoint) |
| `sex` | Sex (1 = Male, 2 = Female) |
| `baseline_egfr_23` | Baseline eGFR (ml/min/1.73 m²) |
| `feature_XXXX` × 6,592 | Log₂-normalised, ComBat-corrected SomaScan plasma protein abundances |

All protein features have been **batch-corrected** using ComBat so they are directly comparable across the training and test cohorts.

---

## Environment Setup

One person per team should be responsible for creating and managing the team's virtual environment.

**First-time setup:**

```bash
module load medaihack/spring-2026
module load python3/3.12.4

# Replace YOUR_TEAM and venv_name with your team directory and preferred name
virtualenv /projectnb/medaihack/YOUR_TEAM/venv_name
source /projectnb/medaihack/YOUR_TEAM/venv_name/bin/activate
pip install -r requirements.txt
```

**Activating the environment in subsequent sessions:**

- **Jupyter or Code Server (OnDemand):** Load both modules and place the `source` command in the pre-launch dialog box.
- **Batch scripts:** Include all three commands:

```bash
module load medaihack/spring-2026
module load python3/3.12.4
source /projectnb/medaihack/YOUR_TEAM/venv_name/bin/activate
```

**Verify your setup:**

```bash
python model.py
```

---

## Pipeline

```
BKBC_train/train.csv
       │
       ├──→ evaluate.py    (iterate: k-fold CV on training data)
       │        └── results/cv_results.csv, confusion matrices
       │
       ├──→ train.py        (final: train on ALL data, save weights)
       │        └── weights/xgboost_model.json, feature_cols.json
       │
       └──→ predict.sh      (inference: predict on new samples)
                └── predictions.csv
```

---

## Step 1 — Evaluate with cross-validation

Use this to iterate on your model. It runs stratified k-fold CV on the training data and reports AUC and log loss per fold.

```bash
python evaluate.py --data ../../BKBC_train/train.csv
```

**Output:**
- Per-fold metrics (AUC, log loss)
- Confusion matrix plots
- Summary table comparing XGBoost and Lasso LR

---

## Step 2 — Train final model

Once you are happy with your model, train on ALL the training data and save weights:

```bash
python train.py --data ../../BKBC_train/train.csv
```

This saves `weights/xgboost_model.json` and `weights/feature_cols.json`.

---

## Step 3 — Predict on new data

```bash
bash predict.sh /path/to/new_data.csv
# or with custom output path:
bash predict.sh /path/to/new_data.csv my_predictions.csv
```

If the input file contains an `ati` column, evaluation metrics are printed automatically.

**Output columns:**

| Column | Description |
|--------|-------------|
| `sample_id` | Patient identifier |
| `prob_ati` | Predicted probability of ATI (0–1) |
| `pred_label` | Hard prediction: 0 = No ATI, 1 = ATI |
| `true_label` | Ground truth (only if `ati` column is present) |

---

## File Structure

```
BKBC/
├── predict.sh         ← bash wrapper for prediction
├── predict.py         ← inference on new data
├── train.py           ← train model, save weights
├── evaluate.py        ← k-fold cross-validation
├── model.py           ← model definitions and constants
├── preprocess.py      ← data loading helpers
├── requirements.txt   ← Python dependencies
├── README.md          ← this file
└── weights/           ← pre-trained model (ready to use)
    ├── xgboost_model.json
    └── feature_cols.json
```

---

## Baseline Performance

The provided XGBoost baseline achieves approximately:

| Metric | BKBC 5-Fold CV |
|--------|----------------|
| AUC    | ~0.76          |
| Log loss | ~0.59        |

---

## Tips for Improving Beyond the Baseline

1. **Feature selection** — 6,592 features is high-dimensional. Use LASSO coefficients, univariate tests, or biological priors to select a focused subset.

2. **Hyperparameter tuning** — try `optuna` or `GridSearchCV` on `n_estimators`, `max_depth`, `colsample_bytree`, or the LR regularisation strength `C`.

3. **Alternative models** — random forests, ElasticNet, neural networks, or ensemble stacking.

4. **Clinical-only baseline** — train on `age`, `sex`, `baseline_egfr_23` alone to quantify the protein contribution.

5. **Class imbalance** — experiment with `scale_pos_weight` in XGBoost or `class_weight='balanced'` in scikit-learn.

---

## Getting Help

```bash
python train.py    --help
python evaluate.py --help
python predict.py  --help
```
