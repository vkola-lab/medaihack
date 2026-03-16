# VI-LUAD Starter Code

This starter code gives you a working end-to-end pipeline for the VI-LUAD classification task. Run it as-is to get a baseline, then explore and improve specific steps. Refer to the provided slide deck for task background, data description, and evaluation criteria.

---

## Labels

The primary task is binary classification — VITUMOR vs. NONVITUMOR:

| Label | Index | Description |
|-------|-------|-------------|
| `NONVITUMOR` | 0 | Tumor slide from a patient WITHOUT vascular invasion |
| `VITUMOR` | 1 | Tumor slide from a patient WITH vascular invasion |

NONTUMOR slides (normal lung tissue) are excluded from the train/test splits but their patch features are extracted so you can leverage them in creative ways — for example, to help the model learn what tumor vs. non-tumor tissue looks like.

---

## Environment Setup

Each team should create their own conda environment. This avoids package version conflicts across teams and lets you freely install any additional packages your approach requires.

```bash
module load miniconda
conda create -n vi_luad python=3.11 -y
conda activate vi_luad
pip install -r starter_code/requirements.txt
```

`requirements.txt` lists the minimum packages needed to run the provided scripts. You are encouraged to install anything else your approach needs on top of it.

For all subsequent sessions, just activate the environment:

```bash
module load miniconda
conda activate vi_luad
```

To install additional packages (e.g. if your approach needs `transformers` or `einops`):

```bash
pip install <package-name>
```

> **Warning:** Do **not** reinstall or upgrade `torch`, `torchvision`, or any `cuda`-related package. The versions in `requirements.txt` are matched to the CUDA driver on the cluster — upgrading them will likely break GPU support.

---

## Pipeline Overview

```
WSI (.svs files, 709 total)
      │
      ▼
preprocess.py  ──►  features/<slide>.pt    (patch feature tensors, shape N×768)
                    splits/fold_{0-4}.json  (train/test patient-level splits,
                                             VITUMOR/NONVITUMOR only)
      │
      ▼
train_eval.py  ──►  checkpoints/fold_{0-4}.pth          (trained model weights)
                    predictions/fold_{0-4}.json          (per-slide test predictions)
                    predictions/fold_{0-4}_patients.json (per-patient test predictions)
                    console: per-fold metrics + CV summary (slide-level and patient-level)
```

---

## Step 1 — Feature Extraction (`preprocess.py`)

Tiles each WSI into patches, runs each patch through [cTransPath](https://github.com/Xiyue-Wang/TransPath) (a pathology-specific vision transformer) to produce a 768-dim feature vector, and saves 5-fold patient-level cross-validation splits.

**Submit the job:**
```bash
cp starter_code/run_preprocess_example.sh my_preprocess.sh
# Edit my_preprocess.sh: fill in all <placeholder> paths
qsub my_preprocess.sh
```

**Monitor:**
```bash
tail -f preprocess.log
qstat -u $USER
```

**Expected output (abridged):**
```
==========================================
Job started: Sun Mar 15 21:08:25 EDT 2026
Running on host: scc-j13
==========================================
Using device: cuda
  GPU: NVIDIA L40S

Kept 506 VITUMOR/NONVITUMOR slides for splits (203 NONTUMOR slides excluded from
splits but their features will still be extracted for optional use)

Patient-level label distribution:
  NONVITUMOR: 127 patients
  VITUMOR:    118 patients
  Total: 245 patients, 506 slides

Fold 0: train=403 slides (196 patients) | test=103 slides (49 patients)
  train labels: {'NONVITUMOR': 232, 'VITUMOR': 171}
  test  labels: {'NONVITUMOR':  59, 'VITUMOR':  44}
...
Saved fold 0 → splits/fold_0.json
...
Saved fold 4 → splits/fold_4.json

============================================================
STEP: Loading cTransPath feature extractor
============================================================
[1/709] Processing 10987.svs ...
  Found 1453 tissue patches in 10987.svs
  Saved features: shape=torch.Size([1453, 768]) → features/10987.pt
[2/709] Processing 11003.svs ...
  Found 2312 tissue patches in 11003.svs
  Saved features: shape=torch.Size([2312, 768]) → features/11003.pt
...
[709/709] Processing NLSI0000061.svs ...
==========================================
Job finished: Sun Mar 15 22:14:52 EDT 2026
==========================================
```

Each `.pt` file is a PyTorch tensor of shape `(N_patches, 768)` — one row per tissue patch. The number of patches per slide varies depending on tissue area.

---

## Step 2 — Training and Evaluation (`train_eval.py`)

Trains a Multiple Instance Learning (MIL) model on the extracted features and evaluates it with 5-fold cross-validation. Metrics are reported both per-slide and per-patient.

**Patient-level aggregation rule:** a patient is predicted VITUMOR if at least one of their slides is predicted VITUMOR (argmax == 1). For AUC, the patient score is max(P(VITUMOR)) across all slides.

**Submit the job:**
```bash
cp starter_code/run_train_eval_example.sh my_train_eval.sh
# Edit my_train_eval.sh: fill in all <placeholder> paths
qsub my_train_eval.sh
```

**Monitor:**
```bash
tail -f train_eval.log
```

**Expected output (abridged):**
```
==========================================
Job started:  Sun Mar 15 22:37:56 EDT 2026
Host:         scc-j13
Features dir: 709 .pt files available
Splits dir:   5 fold JSON files
==========================================
Device: cuda
  GPU: NVIDIA L40S

============================================================
FOLD 0  —  403 train slides / 103 test slides
============================================================

[Data loading]
  [SlideDataset] Ready: 403 slides from 'starter_code/features'.
  [SlideDataset] Ready: 103 slides from 'starter_code/features'.
MILClassifier (mean pooling) built: 197,378 trainable parameters
  feature_dim=768  hidden_dim=256  num_classes=2  dropout=0.25

[Training]  epochs=20  lr=0.0001  weight_decay=0.0001  batch_size=1
  (printing test metrics every 5 epochs)
  Epoch   1/20  train_loss=0.6782
  ...
  Epoch   5/20  train_loss=0.6650  test_logloss=0.7021  test_auc=0.5810
  ...
  Epoch  20/20  train_loss=0.6461  test_logloss=0.6730  test_auc=0.5917

[Evaluation]  Loading best checkpoint (train_loss=0.6461)

  [Per-slide]
    Log loss : 0.6730
    AUC      : 0.5917

  [Per-patient]
    Log loss : 0.6373
    AUC      : 0.6632
    Accuracy : 0.6122

  Checkpoint saved → checkpoints/fold_0.pth
  Per-slide predictions  saved → predictions/fold_0.json
  Per-patient predictions saved → predictions/fold_0_patients.json

...
(folds 1–4 follow the same format)
...

========================================================================
CROSS-VALIDATION SUMMARY
========================================================================

  [Per-slide]
    Fold    Log Loss       AUC
  ------  ----------  --------
       0      0.6730    0.5917
       1      0.6321    0.7308
       2      0.6502    0.6004
       3      0.6601    0.6726
       4      0.7255    0.5952
  ------  ----------  --------
    mean      0.6682    0.6381
     std      0.0316    0.0552

  [Per-patient]
    Fold    Log Loss       AUC    Accuracy
  ------  ----------  --------  ----------
       0      0.6373    0.6632      0.6122
       1      0.6228    0.7088      0.6939
       2      0.6524    0.6316      0.6122
       3      0.6258    0.7368      0.6327
       4      0.6523    0.6667      0.5918
  ------  ----------  --------  ----------
    mean      0.6381    0.6814      0.6286
     std      0.0126    0.0370      0.0351

Done. Checkpoints saved to: starter_code/checkpoints
      Predictions saved to:  starter_code/predictions
==========================================
Job finished: Sun Mar 15 22:41:26 EDT 2026
==========================================
```

The baseline achieves ~0.68 patient-level AUC. NONVITUMOR patients are consistently harder to classify correctly — this is the primary failure mode and a natural starting point for improvement.

Each `predictions/fold_{i}.json` contains one entry per test slide with `filename`, `pid`, `true_label`, `pred_label`, and per-class probabilities (`NONVITUMOR`, `VITUMOR`).

Each `predictions/fold_{i}_patients.json` contains one entry per test patient with `pid`, `true_label`, `pred_label`, `patient_score` (max P(VITUMOR) across slides), and `n_slides`.

---

## Directory Structure

After running both steps your directory should look like:

```
<your_project_dir>/
├── starter_code/
│   ├── preprocess.py              # feature extraction + splits
│   ├── model.py                   # MIL model definition
│   ├── train_eval.py              # training + evaluation loop
│   ├── ctran.py                   # cTransPath architecture (do not modify)
│   ├── run_preprocess_example.sh
│   ├── run_train_eval_example.sh
│   ├── requirements.txt
│   └── README.md
├── features/
│   ├── 10987.pt                   # (N_patches, 768) tensor per slide
│   └── ...
├── splits/
│   ├── fold_0.json                # {"train": [...], "test": [...]}
│   └── ...  fold_4.json
├── checkpoints/
│   ├── fold_0.pth                 # best model checkpoint per fold
│   └── ...  fold_4.pth
└── predictions/
    ├── fold_0.json                # per-slide test predictions for fold 0
    ├── fold_0_patients.json       # per-patient test predictions for fold 0
    └── ...  fold_4.json / fold_4_patients.json
```

---

## Tips for Improvement

The baseline is intentionally simple. Here are directions worth exploring:

- **Better patch features**: swap in a stronger pretrained encoder (e.g. a larger pathology foundation model). This is likely the highest-leverage change. Tuning `--patch_size` or `--read_level` is also possible but requires re-running the full preprocessing pipeline, which is time-consuming — only worth it if you have a specific reason.
- **Better aggregation**: the baseline uses mean pooling (all patches weighted equally) — consider attention pooling, clustering, graph-based methods, or transformer-based aggregation over patches.
- **Class imbalance**: NONVITUMOR (127 patients) slightly outnumbers VITUMOR (118 patients). Try class-weighted loss or label smoothing if your model skews toward one class.
- **Longer training**: the baseline runs only 20 epochs with no scheduler. A cosine annealing schedule may help. For early stopping, consider adding a validation split — you can do this by modifying `create_patient_splits` in `preprocess.py` to produce train/val/test splits instead of train/test.
- **Leveraging NONTUMOR slides**: 203 NONTUMOR slides have features available in `features/`. These are excluded from the train/test splits but can be used in creative ways — for example, as additional negative examples or to help the model learn discriminative tumor features.

For a full list of configurable options, run:
```bash
python starter_code/preprocess.py --help
python starter_code/train_eval.py --help
```
