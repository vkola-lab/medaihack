# MedAIhack submission VI-LUAD Breaking Bad
**Team 15 - Breaking Bad**

Members:
- Sanjiv Sridhar
- Tarushi Gandhi
- Jigar Kanakhara
- Aum Ghelani
- Harsha B Beth

# Project summary:

We built a patient-level Multiple Instance Learning pipeline for vascular invasion prediction using pre-extracted UNI2-h pathology patch embeddings. We added 2D sinusoidal positional encodings to incorporate slide spatial information, projected features into a lower-dimensional space, and trained an ACMIL model with five gated-attention branches. During training, we used Stochastic Top-K Instance Masking to prevent attention collapse and added Attention Entropy Maximization to encourage the model to focus on diverse informative regions. The loss combined patient-level max BCE, branch-wise cross-entropy, label smoothing, class weighting, and entropy regularization. We used patient-level stratified 5-fold cross-validation, held-out validation for early stopping and temperature scaling, and trained five seeds per fold. Final predictions came from a 25-model ensemble with calibrated and clipped probabilities.


## File descriptions and setup instructions same as the starter code given below:




# VI-LUAD Starter Code

This starter code gives you a working end-to-end pipeline for the VI-LUAD classification task. Run it as-is to get a baseline, then explore and improve specific steps. Refer to the provided slide deck for task background, data description, and evaluation criteria.

**Evaluation:** Your final model will be evaluated on external test datasets from 2 different institutions using **per-patient log loss** as the scoring metric. See Step 2 for how patient-level predictions are aggregated and Step 3 for how to submit.

---

## Labels

The primary task is binary classification - VITUMOR vs. NONVITUMOR:

| Label | Index | Description |
|-------|-------|-------------|
| `NONVITUMOR` | 0 | Tumor slide from a patient WITHOUT vascular invasion |
| `VITUMOR` | 1 | Tumor slide from a patient WITH vascular invasion |

NONTUMOR slides (normal lung tissue) are excluded from the train/test splits but their patch features are extracted so you can leverage them in creative ways, e.g. to help the model learn what tumor vs. non-tumor tissue looks like.

---

## Getting Started

Copy the `starter_code` directory to your team's project directory and work from there. All paths in the scripts are relative to your project directory.

```bash
cp -r /projectnb/medaihack/VI_LUAD/starter_code /projectnb/medaihack/YOUR_TEAM/
cd /projectnb/medaihack/YOUR_TEAM
```

Run all subsequent commands from `/projectnb/medaihack/YOUR_TEAM`.

---

## Environment Setup

One person per team should create the team's virtual environment. All team members then activate it for their sessions.

### First-time setup

```bash
module load medaihack/spring-2026
module load python3/3.12.4
virtualenv /projectnb/medaihack/YOUR_TEAM/vi_luad
source /projectnb/medaihack/YOUR_TEAM/vi_luad/bin/activate
pip install -r starter_code/requirements.txt
```

The `medaihack/spring-2026` module sets cache directories (`HF_HOME`, `TORCH_HOME`, etc.) to `/projectnb/medaihack/.cache/$USER` so they don't fill your home directory.

`requirements.txt` lists the minimum packages needed to run the provided scripts. You are encouraged to install anything else your approach needs on top of it.

### Subsequent sessions

For terminal and batch scripts, include these three lines:

```bash
module load medaihack/spring-2026
module load python3/3.12.4
source /projectnb/medaihack/YOUR_TEAM/vi_luad/bin/activate
```

For **OnDemand** (Jupyter or Code Server): load the two modules in the module list, and place the `source` command in the pre-launch dialog box.

To install additional packages (e.g. if your approach needs `transformers` or `einops`):

```bash
pip install <package-name>
```

> **Warning:** Do **not** reinstall or upgrade `torch`, `torchvision`, or any `cuda`-related package. These versions are matched to the cluster and scoring environment. Your model will fail to be scored if you use different versions.

---

## Pipeline Overview

```
Pre-extracted UNI2-h features (dict with "features" N×1536 and "coords" N×2, one .pt file per slide)
      │   /projectnb/medaihack/VI_LUAD_Project/WSI_Data/processed/
      │
      ▼
create_splits.py  ──►  splits/fold_{0-4}.json   (train/test patient-level splits)
      │
      ▼
train_eval.py     ──►  checkpoints/fold_{0-4}.pth          (trained model weights)
                       predictions/fold_{0-4}.json          (per-slide test predictions)
                       predictions/fold_{0-4}_patients.json (per-patient test predictions)
                       console: per-fold metrics + CV summary (slide-level and patient-level)
```

Features have been pre-extracted from all 709 slides using [UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h) (ViT-Giant pretrained on large-scale pathology images) and are ready to use. Each `.pt` file is a Python dict with two tensors:
- `data["features"]`: shape `(N_patches, 1536)`, one UNI2-h feature vector per tissue patch
- `data["coords"]`: shape `(N_patches, 2)`, ordinal `(col, row)` grid index of each patch (adjacent patches differ by 1). The baseline model does not use coordinates.

The number of patches per slide varies with tissue area. The baseline model uses 506 of these slides (VITUMOR and NONVITUMOR only); the remaining 203 NONTUMOR slides have features available for optional use.

---

## Step 1: Create Data Splits (`create_splits.py`)

Reads the clinical label file, groups slides by patient, and creates 5-fold stratified patient-level cross-validation splits. You must run this before training.

**Why patient-level?** The same patient may contribute multiple slides. Splitting at the slide level would allow slides from the same patient to appear in both train and test, which is a form of data leakage. Patient-level splits prevent this.

**Submit the job:**
```bash
# Edit the script: fill in YOUR_TEAM with your team's directory name
bash starter_code/run_create_splits_example.sh
```

Since no GPU is required, this completes in seconds. You can also run it directly in a terminal:
```bash
python starter_code/create_splits.py
```

**Expected output (abridged):**
```
============================================================
STEP: Loading label file
============================================================
Loaded 506 VITUMOR/NONVITUMOR slides

============================================================
STEP: Creating cross-validation splits
============================================================

Patient-level label distribution:
  NONVITUMOR: 150 patients
  VITUMOR: 95 patients
  Total: 245 patients, 506 slides

Fold 0: train=403 slides (196 patients) | test=103 slides (49 patients)
  train labels: {'NONVITUMOR': 229, 'VITUMOR': 174}
  test  labels: {'NONVITUMOR': 63, 'VITUMOR': 40}
...

============================================================
STEP: Saving splits
============================================================
Saved fold 0 → starter_code/splits/fold_0.json
...
Saved fold 4 → starter_code/splits/fold_4.json

Done! Splits saved to: starter_code/splits
```

---

## Step 2: Training and Evaluation (`train_eval.py`)

Trains a Multiple Instance Learning (MIL) model on the pre-extracted UNI2-h features and evaluates it with 5-fold cross-validation. Metrics are reported both per-slide and per-patient.

**Patient-level aggregation rule:** a patient is predicted VITUMOR if at least one of their slides is predicted VITUMOR (argmax == 1). For AUC, the patient score is max(P(VITUMOR)) across all slides.

### Point to the pre-extracted features

When submitting your job, pass:
```
--features_dir /projectnb/medaihack/VI_LUAD_Project/WSI_Data/processed
--splits_dir   starter_code/splits
```

### Submit the job

```bash
# Edit the script: fill in YOUR_TEAM with your team's directory name
bash starter_code/run_train_eval_example.sh
```

**Monitor:**
```bash
tail -f train_eval.log
```

**Expected output (abridged):**
```
Device: cuda
  GPU: NVIDIA L40S

============================================================
FOLD 0  —  403 train slides / 103 test slides
============================================================

[Data loading]
  [SlideDataset] Ready: 403 slides from '/projectnb/medaihack/VI_LUAD_Project/WSI_Data/processed'.
  [SlideDataset] Ready: 103 slides from '/projectnb/medaihack/VI_LUAD_Project/WSI_Data/processed'.
MILClassifier (mean pooling) built: 393,986 trainable parameters
  feature_dim=1536  hidden_dim=256  num_classes=2  dropout=0.25

[Training]  epochs=20  lr=0.0001  weight_decay=0.0001  batch_size=1
  (printing test metrics every 5 epochs)
  Epoch   1/20  train_loss=0.6740
  ...
  Epoch   5/20  train_loss=0.6074  test_logloss=0.6098  test_auc=0.6782
  ...
  Epoch  20/20  train_loss=0.5046  test_logloss=0.6591  test_auc=0.6357

[Evaluation]  Loading best checkpoint (train_loss=0.5046)

  [Per-slide]
    Log loss : 0.6591
    AUC      : 0.6357

  [Per-patient]
    Log loss : 0.6888
    AUC      : 0.6211
    Accuracy : 0.5102

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
       0      0.6591    0.6357
       1      0.8450    0.5696
       2      0.5912    0.7338
       3      0.8079    0.5946
       4      0.7373    0.6157
  ------  ----------  --------
    mean      0.7281    0.6299
     std      0.0934    0.0564

  [Per-patient]
    Fold    Log Loss       AUC    Accuracy
  ------  ----------  --------  ----------
       0      0.6888    0.6211      0.5102
       1      0.7925    0.6000      0.5306
       2      0.6263    0.7316      0.6531
       3      0.6534    0.6632      0.6939
       4      0.7252    0.6404      0.6122
  ------  ----------  --------  ----------
    mean      0.6972    0.6512      0.6000
     std      0.0581    0.0453      0.0702

Done. Checkpoints saved to: starter_code/checkpoints
      Predictions saved to:  starter_code/predictions
==========================================
Job finished: Wed Apr  1 23:19:37 EDT 2026
==========================================
```

The baseline achieves ~0.65 patient-level AUC. NONVITUMOR patients are consistently harder to classify correctly. This is the primary failure mode and a natural starting point for improvement.

Each `predictions/fold_{i}.json` contains one entry per test slide with `filename`, `pid`, `true_label`, `pred_label`, and per-class probabilities (`NONVITUMOR`, `VITUMOR`).

Each `predictions/fold_{i}_patients.json` contains one entry per test patient with `pid`, `true_label`, `pred_label`, `patient_score` (max P(VITUMOR) across slides), and `n_slides`.

---

## Step 3: Leaderboard Submission (`predict.sh` + `predict.py`)

Your model's leaderboard score is determined by per-patient log loss on a held-out external test set from 2 different institutions. The organizers will run `predict.sh` on your behalf - you do not need to run it yourself (you will not have access to the test data).

**What you need to do:**

1. Open `predict.sh` and fill in:
   - `TEAM` - your team's directory name (same as YOUR_TEAM above)
   - `CHECKPOINT` - path to your model checkpoint file

2. If you used the baseline model as-is, no further changes are needed.

3. If you changed the model architecture, open `predict.py` and update `load_checkpoint()` (Section 1) to import and instantiate your model correctly. Your model must:
   - Accept a feature tensor of shape `(N, 1536)` as input
   - Return logits of shape `(1, 2)` as the first element of the output tuple

Do not modify Sections 2-6 of `predict.py`.

**Sanity check:** Before submitting, verify that your virtual environment has the correct PyTorch version:

```bash
python -c "import torch; print(torch.__version__)"
```

This should print `2.8.0`. If it does not, your model will fail to be scored.

---

## Directory Structure

After training your directory should look like:

```
/projectnb/medaihack/YOUR_TEAM/
├── starter_code/
│   ├── create_splits.py           # patient-level cross-validation splits
│   ├── model.py                   # MIL model definition
│   ├── train_eval.py              # training + evaluation loop
│   ├── splits/
│   │   ├── fold_0.json            # {"train": [...], "test": [...]}
│   │   └── ...  fold_4.json
│   ├── run_create_splits_example.sh
│   ├── run_train_eval_example.sh
│   ├── predict.py                 # leaderboard inference (modify Section 1 if needed)
│   ├── predict.sh                 # leaderboard submission (fill in TEAM + CHECKPOINT)
│   ├── requirements.txt
│   └── README.md
├── checkpoints/
│   ├── fold_0.pth                 # best model checkpoint per fold
│   └── ...  fold_4.pth
└── predictions/
    ├── fold_0.json                # per-slide test predictions for fold 0
    ├── fold_0_patients.json       # per-patient test predictions for fold 0
    └── ...  fold_4.json / fold_4_patients.json
```

Pre-extracted features (read-only, shared across teams):
```
/projectnb/medaihack/VI_LUAD_Project/WSI_Data/processed/
    ├── 10987.pt                   # dict: "features" (N_patches, 1536), "coords" (N_patches, 2)
    └── ...
```

---

## Visualizing Whole Slide Images with QuPath

If you want to view the original whole slide images, you can use QuPath on the SCC:

1. Start an OnDemand Desktop with 4 cores for viewing the files. Don't select a GPU if this is for viewing only. Have it load the `qupath/0.5.1` module.
2. When the desktop opens, in the terminal change to the image directory:
   ```bash
   cd /projectnb/medaihack/VI_LUAD_Project/WSI_Data/wsi
   ```
3. Start QuPath:
   ```bash
   QuPath
   ```
4. When the GUI opens, click **File → Open**. Select a `.svs` image that you'd like to see.
5. When finished, click **File → Quit** to close QuPath.

---

## Tips for Improvement

The baseline is intentionally simple. Here are directions worth exploring:

- **Better aggregation**: the baseline uses mean pooling (all patches weighted equally). Consider attention pooling, clustering, or transformer-based aggregation over patches.
- **Spatial models**: each `.pt` file includes `data["coords"]`, the ordinal `(col, row)` grid position of every patch. These coordinates can be used to build a spatial graph between neighboring patches (e.g., with PyTorch Geometric) or to add positional encodings before aggregation.
- **Class imbalance**: NONVITUMOR (150 patients) outnumbers VITUMOR (95 patients). Try class-weighted loss or label smoothing if your model skews toward one class.
- **Longer training**: the baseline runs only 20 epochs with no scheduler. A cosine annealing schedule may help. For early stopping, consider adding a validation split by modifying `create_patient_splits` in `create_splits.py` to produce train/val/test splits instead of train/test.
- **Leveraging NONTUMOR slides**: 203 NONTUMOR slides have features available in the shared features directory. These are excluded from the train/test splits but can be used in creative ways, e.g. as additional negative examples or to help the model learn discriminative tumor features.

For a full list of configurable options, run:
```bash
python starter_code/create_splits.py --help
python starter_code/train_eval.py --help
```
