# Amyloid PET Centiloid Prediction Challenge

## Overview

Predict **centiloid scores** from preprocessed 3D amyloid PET brain scans. Centiloid is a standardized quantitative measure of amyloid-beta plaque burden in the brain and is a key biomarker for Alzheimer's disease. Higher centiloid values indicate greater amyloid deposition.

**Task:** Given a preprocessed 3D PET volume and the radiotracer used, predict the continuous centiloid score (regression).

## Setup

```bash
conda create -n abpet python=3.11 -y
conda activate abpet
pip install -r requirements.txt
```

## Data

Cohort | N Samples | Description                               |
-------|-----------|-------------------------------------------|
NACC   | 1,500     | National Alzheimer's Coordinating Center  |
A4     | 1,000     | Anti-Amyloid Treatment in Asymptomatic AD |

Each sample is a preprocessed `.npy` file with an associated centiloid score and tracer label.

### CSV Format

| Column      | Type   | Description                                            |
|-------------|--------|--------------------------------------------------------|
| `npy_path`  | str    | Path to the preprocessed `.npy` file                   |
| `CENTILOIDS`| float  | Target — amyloid burden score (typically 0–150+)       |
| `TRACER.AMY`| str    | Radiotracer used (e.g., Florbetapir, Florbetaben, PIB) |
| `ID`        | str    | Subject identifier                                     |

### Image Format

Each `.npy` file contains a single preprocessed PET volume:

- **Shape:** `(1, 128, 128, 128)` — 1 channel, 128³ voxels
- **Dtype:** `float32`
- **Value range:** `[0, 1]` (min-max normalized)

### Why Tracer Matters

Different radiotracers (Florbetapir, Florbetaben, PIB, etc.) bind to amyloid with different affinities and produce different uptake patterns in PET images. The centiloid scale was designed to harmonize across tracers, but the raw images still differ by tracer. Your model should account for this — a tracer embedding is one common approach.

## Preprocessing Already Applied

All images have been preprocessed from raw NIfTI PET scans. The following transformations were applied **in order** (you do **not** need to redo any of these):

### 1. Ensure Channel First
Converts the loaded nifti image to `(C, H, W, D)` format, where `C` is the channel dimension.

### 3. Orientation o RAS
Reorients the image to **RAS** (Right-Anterior-Superior) standard neuroimaging orientation. This ensures consistent spatial alignment across subjects and scanners.

### 4. Isotropic Resampling to 2mm
Resamples the image to **2mm × 2mm × 2mm isotropic voxel spacing** using trilinear interpolation. Raw PET scans vary widely in resolution across scanners and protocols — this standardizes the spatial scale.

### 5. Foreground Cropping
Removes background voxels (air/zero-padding outside the brain) with a 10-voxel margin using MONAI's `CropForeground`. This reduces unnecessary empty space.

### 6. Resize to 128³
Resizes the cropped volume to a fixed `128 × 128 × 128` spatial size using trilinear interpolation. This guarantees uniform input dimensions for the network.

### 7. Spatial Padding
Pads to exactly `128 × 128 × 128` if the resize output is slightly smaller (safety step).

### 8. Dynamic Frame Averaging
Some PET scans have multiple temporal frames (dynamic acquisitions). If multiple frames exist, they are averaged into a single static volume. This produces a single `(1, 128, 128, 128)` output.

### 9. Shape Enforcement
A final safety check centers and crops/pads the volume to ensure the exact output shape `(1, 128, 128, 128)`.

### 10. Min-Max Normalization to [0, 1]
Each volume is independently normalized to the `[0, 1]` range:

```
img = (img - img.min()) / (img.max() - img.min())
```

## Project Structure

```
ABPET/
├── data/
│   ├── npy_files/       # All .npy volumes
│   ├── train.csv        # Training split
│   └── val.csv          # Validation split
├── code/
│   ├── model.py         # 3D CNN architecture
│   ├── train.py         # Training script
│   ├── predict.py       # Inference script
│   ├── dataset.py       # Shared dataset class
│   └── losses.py        # Loss functions
├── README.md
└── requirements.txt
```

## Getting Started

```bash
# Create conda environment
conda create -n abpet python=3.11 -y
conda activate abpet
pip install -r requirements.txt 

# Train
python train.py --train_csv ../data/train.csv --val_csv ../data/val.csv

# Predict
python predict.py --csv ../data/val.csv --checkpoint best_model.pt
```

## Evaluation

Models will be evaluated on the validation set using:
- **Primary metric:** Mean Absolute Error (MAE) in centiloid units
- **Secondary:** Pearson correlation coefficient between predicted and true centiloid scores

## Tips

- The centiloid distribution is often skewed (many low values, fewer high values). Consider how your loss function handles this.
- Tracer conditioning can be important — consider integrating the `TRACER.AMY` column.
- The data is already normalized to [0, 1], so you can feed it directly into your network.
- 3D medical images are memory-intensive. Watch your batch size.