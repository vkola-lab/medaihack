# Challenge overview

Alzheimer's disease (AD) is the most common form of dementia and a growing global health crisis. One of its earliest hallmarks is the buildup of abnormal protein deposits in the brain called amyloid plaques, which can be detected years before symptom onset using Positron Emission Tomography (PET) imaging. The Centiloid scale provides us with a standardized measure of amyloid burden in brain amyloid PET scans, with scores near zero or negative indicating little to no amyloid, and higher scores signaling significant accumulation associated with AD risk. In this challenge, you will build computer vision models to predict Centiloid scores from already preprocessed 3D amyloid PET scans. However, these brain scans were acquired using different PET radiotracers since different hospitals and studies tend to use different imaging protocols. While all tracers are radioactive compounds that bind to amyloid plaques, each of them produces images with different intensity profiles and noise characteristics. Your models should be robust to this variation, and encoding tracer identity as part of your approach is encouraged, though not required. Participants will have access to 2,000 training samples and 500 validation samples, with final rankings determined on a held-out test set evaluated by the judges. Strong solutions may be included as contributions in a publishable research paper.

# Amyloid PET Centiloid Prediction Challenge

This starter code provides a working end-to-end pipeline for the **Amyloid PET Centiloid Prediction Challenge**. You can run it as-is to obtain a baseline, then improve specific components such as the 3D CNN backbone, tracer conditioning, loss function, and training strategy. Refer to the challenge description for task background, data format, preprocessing details, and evaluation criteria.

**Evaluation:** Your model will be evaluated on the validation set using **Mean Absolute Error (MAE)** in centiloid units as the primary metric, with **Pearson correlation coefficient** as the secondary metric. Since the task is continuous centiloid prediction from preprocessed 3D PET volumes, this pipeline is designed as a **regression** framework rather than classification.

## Overview

Predict **centiloid scores** from preprocessed 3D amyloid PET brain scans. Centiloid is a standardized quantitative measure of amyloid-beta plaque burden in the brain and is a key biomarker for Alzheimer's disease. Higher centiloid values indicate greater amyloid deposition.

**Task:** Given a preprocessed 3D PET volume and the radiotracer used, predict the continuous centiloid score.

## Clone GitHub repository

```bash
cd <your_team_folder>
git clone https://github.com/vkola-lab/medaihack.git
cd medaihack/ABPET
```

## Environment Setup

One person per team should create the team's virtual environment. All team members then activate it for their sessions.

### First-time setup

```bash
module load medaihack/spring-2026
module load python3/3.12.4

# Replace YOUR_TEAM and venv_name with your team directory and preferred name
virtualenv /projectnb/medaihack/YOUR_TEAM/venv_name
source /projectnb/medaihack/YOUR_TEAM/venv_name/bin/activate
pip install -r requirements.txt
```

### Subsequent sessions

For terminal and batch scripts, include these three lines:

```bash
module load medaihack/spring-2026
module load python3/3.12.4
source /projectnb/medaihack/YOUR_TEAM/venv_name/bin/activate
```

For **OnDemand** (Jupyter or Code Server): load the two modules in the module list, and place the `source` command in the pre-launch dialog box.

To use your venv as a Jupyter kernel, run this once after activating it:

```bash
python -m ipykernel install --user --name venv_name --display-name "Python (venv_name)"
```

Then refresh JupyterLab and select **Python (venv_name)** from the kernel list.

To install additional packages (e.g. if your approach needs `transformers` or `einops`):

```bash
pip install <package-name>
```

> **Warning:** Do **not** reinstall or upgrade `torch`, `torchvision`, or any `cuda`-related package. The versions in `requirements.txt` are matched to the CUDA driver on the cluster. Upgrading them will likely break GPU support.

---

## Data

```text
/projectnb/medaihack/ABPET/
└── data/
    ├── npy_files/               # All .npy volumes
    ├── train.csv                # Training split (2,000 samples)
    └── val.csv                  # Validation split (500 samples)
```

| Split      | Cohorts   | N Samples | Description                                              |
| ---------- | --------- | --------- | -------------------------------------------------------- |
| Training   | NACC + A4 | 2,000     | 1,195 NACC + 805 A4 samples                              |
| Validation | NACC + A4 | 500       | 305 NACC + 195 A4 samples                                |

Each sample is a preprocessed `.npy` file with an associated centiloid score and tracer label.

### CSV Format

| Column       | Type  | Description                                            |
| ------------ | ----- | ------------------------------------------------------ |
| `npy_path`   | str   | Path to the preprocessed `.npy` file                   |
| `CENTILOIDS` | float | Target — amyloid burden score (typically 0–150+)       |
| `TRACER.AMY` | str   | Radiotracer used: `FBP`, `FBB`, `NAV`, `PIB`           |
| `ID`         | str   | Subject identifier                                     |

### Image Format

Each `.npy` file contains a single preprocessed PET volume:

* **Shape:** `(1, 128, 128, 128)` — 1 channel, 128³ voxels
* **Dtype:** `float32`
* **Value range:** `[0, 1]` (min-max normalized)

### Why Tracer Matters

The four tracers in this dataset are:

| Code  | Full name      |
| ----- | -------------- |
| `FBP` | Florbetapir    |
| `FBB` | Florbetaben    |
| `NAV` | Florbetanav    |
| `PIB` | Pittsburgh Compound B |

Each binds to amyloid with different affinity and produces different uptake patterns. The centiloid scale was designed to harmonize across tracers, but the raw images still differ by tracer. Your model should account for this — a tracer embedding is one common approach.

## Preprocessing Already Applied

All images have been preprocessed from raw NIfTI PET scans. The following transformations were applied **in order** (you do **not** need to redo any of these):

### 1. Ensure Channel First

Converts the loaded NIfTI image to `(C, H, W, D)` format, where `C` is the channel dimension.

### 2. Orientation to RAS

Reorients the image to **RAS** (Right-Anterior-Superior) standard neuroimaging orientation. This ensures consistent spatial alignment across subjects and scanners.

### 3. Isotropic Resampling to 2mm

Resamples the image to **2mm × 2mm × 2mm isotropic voxel spacing** using trilinear interpolation. Raw PET scans vary widely in resolution across scanners and protocols — this standardizes the spatial scale.

### 4. Foreground Cropping

Removes background voxels (air/zero-padding outside the brain) with a 10-voxel margin using MONAI's `CropForeground`. This reduces unnecessary empty space.

### 5. Resize to 128³

Resizes the cropped volume to a fixed `128 × 128 × 128` spatial size using trilinear interpolation. This guarantees uniform input dimensions for the network.

### 6. Spatial Padding

Pads to exactly `128 × 128 × 128` if the resize output is slightly smaller (safety step).

### 7. Dynamic Frame Averaging

Some PET scans have multiple temporal frames (dynamic acquisitions). If multiple frames exist, they are averaged into a single static volume. This produces a single `(1, 128, 128, 128)` output.

### 8. Shape Enforcement

A final safety check centers and crops/pads the volume to ensure the exact output shape `(1, 128, 128, 128)`.

### 9. Min-Max Normalization to [0, 1]

Each volume is independently normalized to the `[0, 1]` range:

```python
img = (img - img.min()) / (img.max() - img.min())
```

## Repository Structure

```text
/projectnb/medaihack/YOUR_TEAM/medaihack/ABPET/
├── checkpoints/             # Saved model weights (created at train time)
├── logs/                    # Training log files (created at train time)
├── results/                 # Metrics CSV and plots (created at train time)
├── dataset.py               # Shared dataset class
├── losses.py                # Loss functions
├── model.py                 # 3D CNN architecture
├── predict.py               # Inference script
├── predict.sh               # Evaluation entry point
├── train.py                 # Training script
├── visualize_pet.ipynb      # Notebook for exploring PET volumes
├── README.md
└── requirements.txt
```

## Getting Started

To get started, you can visualize the different images using `visualize_pet.ipynb`.

**Interactive (OnDemand terminal):**

```bash
cd /projectnb/medaihack/YOUR_TEAM/medaihack/ABPET
source /projectnb/medaihack/YOUR_TEAM/venv_name/bin/activate

# Train
python train.py --train_csv /projectnb/medaihack/ABPET/data/train.csv --val_csv /projectnb/medaihack/ABPET/data/val.csv

# Predict
python predict.py --csv /projectnb/medaihack/ABPET/data/val.csv --checkpoint checkpoints/best_model.pt
```

## Pipeline

```text
Preprocessed 3D amyloid PET volumes (.npy)
(shape: 1 x 128 x 128 x 128, float32, normalized to [0,1])
+ metadata from train.csv / val.csv
(columns: npy_path, CENTILOIDS, TRACER.AMY, ID)
            |
            v
dataset.py  --->  load PET volume + tracer label + centiloid target
            |
            v
model.py    --->  3D CNN image encoder
                  + tracer embedding
                  + feature fusion
                  + regression head
            |
            v
losses.py   --->  regression loss
                  (MAE / MSE)
            |
            v
train.py    --->  checkpoints/best_model.pt              (best model weights)
                  logs/train_{timestamp}.log             (training log)
                  results/metrics_{timestamp}.csv        (per-epoch metrics)
                  results/curves_{timestamp}.png         (loss / MAE / r plots)
                  results/val_report_{timestamp}.csv     (best-epoch MAE and r, overall + per tracer)
                  console: train loss, val MAE, Pearson correlation
            |
            v
predict.py  --->  predictions.csv
                  columns: ID, npy_path, TRACER.AMY, PREDICTED_CENTILOIDS
```

## Outputs

After training the following files are created automatically:

| File | Description |
| ---- | ----------- |
| `checkpoints/best_model.pt` | Best model weights (lowest val MAE) |
| `logs/train_{timestamp}.log` | Full training log |
| `results/metrics_{timestamp}.csv` | Per-epoch train loss, val MAE, val r |
| `results/curves_{timestamp}.png` | Loss / MAE / correlation plots |
| `results/val_report_{timestamp}.csv` | MAE and Pearson r for best checkpoint — overall and broken down by tracer |

## Baseline Performance

The unmodified starter code achieves the following on the validation set:

| Tracer | N | MAE (CL) | Pearson r |
| ------ | --- | -------- | --------- |
| **ALL** | 500 | **19.77** | **0.790** |
| FBP | 236 | 19.28 | 0.797 |
| FBB | 114 | 20.04 | 0.804 |
| PIB | 133 | 21.17 | 0.790 |
| NAV | 17  | 13.86 | 0.946 |

Your goal is to beat this baseline. Lower MAE and higher Pearson r are better.

## Submission

Before the deadline, make sure your repository is in order:

1. Your best checkpoint is saved at `checkpoints/best_model.pt`
2. `predict.sh` has your team's venv path hardcoded (replace the `.venv` line)
3. If you changed the model architecture, `predict.py` reflects it (see `# MODEL` markers)
4. Test end-to-end: `bash predict.sh /projectnb/medaihack/ABPET/data/val.csv` should produce `predictions.csv` without errors

The judges will clone your repository and run `predict.sh` against the held-out test set.

## Evaluation

Models will be evaluated on a held-out test set. The judges will run:

```bash
bash predict.sh <test.csv> <checkpoint.pt> predictions.csv
```

To test your own predictions on the validation set:

```bash
cd /projectnb/medaihack/ABPET/medaihack/ABPET
source /projectnb/medaihack/YOUR_TEAM/venv_name/bin/activate
python predict.py --csv /projectnb/medaihack/ABPET/data/val.csv --checkpoint checkpoints/best_model.pt --output predictions.csv
```

This calls `predict.py`, which must output a CSV with a `PREDICTED_CENTILOIDS` column. **If you replace `BaselineCNN` with your own model, you must update the import and model instantiation in `predict.py`** (marked with `# MODEL`). Make sure `predict.sh` points to your best checkpoint.

Scoring metrics:

* **Primary:** Mean Absolute Error (MAE) in centiloid units
* **Secondary:** Pearson correlation coefficient between predicted and true centiloid scores

## Tips

* The centiloid distribution is often skewed (many low values, fewer high values). Consider how your loss function handles this.
* Tracer conditioning can be important — consider integrating the `TRACER.AMY` column.
* The data is already normalized to `[0, 1]`, so you can feed it directly into your network.
* 3D medical images are memory-intensive. Watch your batch size and consider mixed-precision training (`torch.amp`).



