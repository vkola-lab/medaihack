"""
preprocess.py — Image Processing Pipeline for VI-LUAD Hackathon
================================================================
This script implements a baseline preprocessing pipeline for whole-slide images (WSIs).
You don't need to read through the whole script — just run it as-is to generate the
features and splits you need, then move on to model.py and run.py. That said, do read
this docstring. It explains the workflow and how to run the script. Come back to the
code itself if you want to improve a specific step. You are encouraged to do so —
better preprocessing can directly boost your model's performance.

The baseline preprocessing pipeline consists of the following steps:
  1. Tissue-background segmentation using HSV color filtering
  2. Tissue tiling into fixed-size patches
  3. Patch feature extraction using cTransPath (a pathology-specific vision transformer)
  4. 5-fold stratified (by VITUMOR/NONVITUMOR label) patient-level cross-validation splits

The primary task is binary classification: VITUMOR vs NONVITUMOR.
  - VITUMOR    (label 1): tumor slide from a VI-positive patient
  - NONVITUMOR (label 0): tumor slide from a VI-negative patient
The baseline splits include only VITUMOR and NONVITUMOR slides. NONTUMOR slides (normal
lung tissue) have features extracted so participants can leverage them in creative ways
(e.g., pretraining, multi-task learning, or as negative examples).

Key concepts:
  - WSI (Whole-Slide Image): A very large digitized tissue slide, often several GB.
    We cannot feed the entire image into a model, so we tile it into small patches.
  - Patch: A crop of the WSI read at a chosen pyramid level (default: level 1, 10×)
    and resized to 224×224 before being passed to cTransPath.
  - Tissue segmentation: Most of a WSI is empty glass (white background). We detect
    tissue regions using color information so we only extract features from meaningful areas.
  - cTransPath: A vision transformer pretrained on pathology images. It converts each
    224×224 patch into a 768-dimensional feature vector.
  - MIL (Multiple Instance Learning): Our downstream model treats each slide as a "bag"
    of patch features and predicts a slide-level label — more on this in model.py.

Usage:
  Copy starter_code/run_preprocess_example.sh, fill in the <placeholder> paths,
  and submit with:  qsub run_preprocess_example.sh

  This runs feature extraction and cross-validation splits in one step.
  Use --output_dir and --splits_dir to control where outputs are saved.

  If features are already extracted and you only want to redo the splits
  (e.g. with a different --n_folds or --random_seed), pass --splits_only to
  skip the image processing step entirely:
    python starter_code/preprocess.py --splits_only \\
        --label_file /projectnb/medaihack/VI_LUAD_Project/Clinical_Data/hackathon_label.txt \\
        --wsi_dir <your_wsi_dir> --splits_dir <your_splits_dir>

Output:
  <output_dir>/<slide_name>.pt   — PyTorch tensor of shape (N, 768), one per slide
  <splits_dir>/fold_0.json       — Train/val split for fold 0
  ...
  <splits_dir>/fold_4.json       — Train/val split for fold 4
"""

import os
import sys
import csv
import json
import argparse
import warnings
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")  # suppress noisy library warnings

# Thread-local storage so each worker thread keeps its own OpenSlide handle open.
# OpenSlide is not safe to share across threads, but a per-thread handle is safe.
_thread_slides = threading.local()


# =============================================================================
# SECTION 1: TISSUE SEGMENTATION
# =============================================================================

def segment_tissue_hsv(wsi, level: int = 2, sat_thresh: float = 15,
                        val_thresh: float = 230, kernel_size: int = 7):
    """
    Detect tissue regions in a WSI using HSV color thresholding.

    Why HSV?
    --------
    H&E-stained tissue has distinctive purple/pink hues with moderate saturation,
    while the glass background is nearly white (low saturation, high brightness).
    By thresholding on saturation (S) and value/brightness (V), we cleanly separate
    tissue from background without needing a trained model.

    Parameters
    ----------
    wsi        : openslide.OpenSlide object
    level      : pyramid level to read for segmentation. Level 0 is full resolution;
                 higher levels are downsampled. Level 2 is ~16× smaller — fast enough
                 for segmentation without losing tissue structure.
    sat_thresh : minimum saturation (0–255) to be considered tissue. Pixels with
                 saturation below this are background (white/grey glass).
    val_thresh : maximum value/brightness (0–255) for tissue. Very bright pixels
                 (close to 255) are also background.
    kernel_size: size of the morphological closing kernel used to fill small holes
                 in the tissue mask.

    Returns
    -------
    tissue_mask : np.ndarray of shape (H, W) dtype bool — True where tissue is present
    downscale   : factor by which this mask is smaller than level-0 (used later to map
                  patch coordinates back to the full-resolution slide)
    """
    import cv2

    # Read the downsampled thumbnail from the WSI pyramid
    thumb = wsi.read_region((0, 0), level, wsi.level_dimensions[level])
    thumb = np.array(thumb.convert("RGB"))  # RGBA -> RGB numpy array

    # Convert to HSV color space
    thumb_hsv = cv2.cvtColor(thumb, cv2.COLOR_RGB2HSV)
    H, S, V = thumb_hsv[:, :, 0], thumb_hsv[:, :, 1], thumb_hsv[:, :, 2]

    # Tissue: moderate saturation AND not too bright
    tissue_mask = (S > sat_thresh) & (V < val_thresh)

    # Morphological closing: fills gaps/holes in the detected tissue regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    tissue_mask = cv2.morphologyEx(tissue_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    tissue_mask = tissue_mask.astype(bool)

    # The downscale factor tells us how big each pixel in the mask is in the full image
    downscale = wsi.level_downsamples[level]

    return tissue_mask, downscale


def get_tissue_patches(wsi, tissue_mask, downscale, patch_size: int = 1024,
                        step_size: int = 1024, tissue_thresh: float = 0.5):
    """
    Tile the WSI and return coordinates of patches that overlap with tissue.

    We slide a window of `patch_size × patch_size` pixels (at level 0 resolution)
    across the slide with stride `step_size`. For each window, we check what fraction
    of it overlaps with the tissue mask. Only patches with sufficient tissue coverage
    (>= tissue_thresh) are kept. To do this efficiently, we tile the downsampled
    thumbnail (tissue mask) rather than iterating over the full-resolution grid.

    Why tile the thumbnail instead of the full-resolution slide?
    ------------------------------------------------------------
    The tissue mask is a tiny downsampled thumbnail (~5500×5300 px for a typical
    40× slide). Tiling it directly and converting the kept positions back to level-0
    coordinates at the end is far more efficient than iterating over the full-res grid
    and mapping each position into the mask — especially when participants experiment
    with small patch sizes.

    Parameters
    ----------
    wsi           : openslide.OpenSlide object
    tissue_mask   : bool mask from segment_tissue_hsv (at the downsampled level)
    downscale     : downscale factor (mask pixel = downscale full-res pixels)
    patch_size    : patch side length in pixels at level 0
    step_size     : stride in pixels at level 0. Equal to patch_size → no overlap.
    tissue_thresh : minimum fraction of the patch that must be tissue (0–1)

    Returns
    -------
    coords : list of (x, y) tuples — top-left corner of each patch in level-0 pixels
    """
    mask_h, mask_w = tissue_mask.shape

    # Convert patch size and stride to thumbnail (mask) coordinates.
    # All tiling happens on the small thumbnail — fast regardless of patch_size.
    mw = max(1, round(patch_size / downscale))   # patch width  in mask pixels
    mh = max(1, round(patch_size / downscale))   # patch height in mask pixels
    ms = max(1, round(step_size  / downscale))   # stride       in mask pixels

    coords = []
    for my in range(0, mask_h - mh + 1, ms):
        for mx in range(0, mask_w - mw + 1, ms):
            # Tissue fraction: what share of this thumbnail tile is tissue?
            tile = tissue_mask[my: my + mh, mx: mx + mw]
            if tile.mean() >= tissue_thresh:
                # Convert thumbnail position back to level-0 (full-resolution) coords
                coords.append((int(mx * downscale), int(my * downscale)))

    return coords


# =============================================================================
# SECTION 2: FEATURE EXTRACTION WITH CTRANSPATH
# =============================================================================

def load_ctranspath(weights_path: str, device: torch.device):
    """
    Load the cTransPath feature extractor.

    cTransPath is a Swin Transformer (swin_tiny_patch4_window7_224) with a custom
    convolutional stem, pretrained on over 15 million pathology image patches using
    a self-supervised momentum-contrast objective. It maps a 224×224 RGB patch to
    a 768-dimensional feature vector that captures tissue morphology.

    Parameters
    ----------
    weights_path : path to the .pth file with pretrained weights
    device       : torch.device for inference (CPU or CUDA GPU)

    Returns
    -------
    model : cTransPath model in eval mode, ready for feature extraction
    """
    # Import from the local ctran.py in the medaihack directory
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from ctran import ctranspath

    model = ctranspath()
    # The weights file may contain full model state or just the encoder
    state_dict = torch.load(weights_path, map_location=device)
    if "model" in state_dict:
        state_dict = state_dict["model"]
    # Remove 'model.' prefix if present (common packaging artifact)
    state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}

    # timm >= 0.9 changed PatchMerging: norm moved from pre-reduction (shape 4C)
    # to post-reduction (shape 2C). Drop the 6 mismatched downsample keys so
    # load_state_dict(strict=False) can proceed — all 174 transformer block
    # weights (attention, FFN, patch embed) still load correctly.
    model_state = model.state_dict()
    state_dict = {k: v for k, v in state_dict.items()
                  if k not in model_state or v.shape == model_state[k].shape}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"  [cTransPath] Loaded {len(state_dict)} keys "
          f"({len(missing)} missing, {len(unexpected)} unexpected)")

    # Replace the classification head with Identity so forward() bypasses the
    # 1000-dim linear layer. In this timm version the head also contains global
    # average pooling, so forward() will return a spatial feature map (B, H, W, C).
    # extract_features_for_slide() handles the pooling explicitly.
    model.head = torch.nn.Identity()

    model.eval()
    model.to(device)
    return model


def get_patch_transforms():
    """
    Return the image preprocessing pipeline for cTransPath.

    cTransPath was trained on patches normalized to ImageNet statistics.
    We apply:
      - Resize to 224×224 (already correct if patch_size=224 but let's be safe)
      - Convert to tensor (scales [0,255] → [0,1])
      - Normalize with ImageNet mean and std (matches training distribution)
    """
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def extract_features_for_slide(wsi_path: str, model, transform, device: torch.device,
                                patch_size: int = 1024, step_size: int = 1024,
                                batch_size: int = 256, num_workers: int = 8,
                                tissue_thresh: float = 0.5, read_level: int = 1):
    """
    Run the full preprocessing pipeline for a single WSI and return patch features.

    Steps:
      1. Open the slide with openslide
      2. Segment tissue using HSV filtering
      3. Generate patch coordinates overlapping with tissue
      4. Read patches in parallel (CPU I/O) and run through cTransPath (GPU) in batches
      5. Return stacked feature matrix

    Patch size and magnification
    -----------------------------
    The default uses 1024×1024 patches at level 0 (40×), resized to 224×224 for
    cTransPath — roughly equivalent to 10× magnification. This is a reasonable
    starting point: large enough to capture tissue structure, few enough patches
    per slide to be practical. You should feel free to experiment. Smaller patches
    (e.g. 512×512 or 256×256) capture finer detail; larger patches or non-square
    crops may work better for certain tasks. Overlapping patches (step_size <
    patch_size) increase coverage at the cost of more computation.

    Why read at a lower pyramid level?
    ------------------------------------
    Each `read_region` call at level 0 (40×) must decompress 16 JPEG tiles to
    assemble a 1024×1024 patch (~88 ms/call on NFS). Level 1 (10×, 4× downsampled)
    stores pre-downsampled tiles: a 256×256 patch covers the same tissue area and
    takes ~11 ms — 8× faster with identical effective magnification for cTransPath.
    Use `read_level` to trade off speed vs. resolution: level 0 for maximum detail,
    level 1 (default) for the best speed/quality balance, level 2 for maximum speed.

    Why parallel reading?
    ---------------------
    `wsi.read_region()` is I/O-bound (disk reads + JPEG tile decompression).
    A ThreadPoolExecutor lets multiple threads overlap disk waits, cutting I/O time
    by roughly num_workers×. Each thread keeps its own OpenSlide handle (thread-safe).

    Parameters
    ----------
    wsi_path    : path to the .svs file
    model       : loaded cTransPath model
    transform   : torchvision transform pipeline (includes Resize(224×224))
    device      : CPU or CUDA
    patch_size  : patch side length in level-0 pixels. Default 1024 works well for
                  40× slides; feel free to try other sizes.
    step_size   : stride between patches in level-0 pixels. Default equals patch_size
                  (non-overlapping); reduce for overlapping patches.
    batch_size  : number of patches per GPU forward pass (reduce if out of memory)
    num_workers : number of parallel threads for patch reading (matches CPU cores)
    read_level  : WSI pyramid level to read patches from (0=40×, 1=10×, 2=2.5×).
                  Level 0 preserves the full 40× resolution but is ~8× slower per
                  read on NFS. Level 1 (default) is a practical starting point;
                  level 0 may yield better features if resolution matters for your
                  approach.

    Returns
    -------
    features : torch.Tensor of shape (N_patches, 768)
               Returns None if no tissue patches are found.
    """
    import openslide

    # Open slide in the main thread for segmentation + coordinate generation only
    wsi = openslide.open_slide(wsi_path)

    # Step 1: tissue segmentation (fast — operates on a small thumbnail)
    tissue_mask, downscale = segment_tissue_hsv(wsi, level=2)

    # Step 2: generate patch coordinates on the full-resolution grid
    coords = get_tissue_patches(wsi, tissue_mask, downscale,
                                patch_size=patch_size, step_size=step_size,
                                tissue_thresh=tissue_thresh)

    # How many pixels to request at read_level to cover patch_size level-0 pixels.
    # e.g. patch_size=1024, level 1 downscale=4 → read_size=256.
    read_level = min(read_level, wsi.level_count - 1)  # clamp to available levels
    read_size = max(1, round(patch_size / wsi.level_downsamples[read_level]))

    wsi.close()  # Close main-thread handle; worker threads open their own below

    if len(coords) == 0:
        print(f"  Warning: no tissue patches found in {Path(wsi_path).name}")
        return None

    print(f"  Found {len(coords)} tissue patches in {Path(wsi_path).name}")

    def read_patch(xy):
        """
        Read one patch using a per-thread OpenSlide handle.

        Each thread opens the slide once and keeps the handle alive for subsequent
        calls (via thread-local storage). This avoids the overhead of opening the
        file repeatedly while remaining thread-safe.
        """
        x, y = xy
        # Open a fresh handle if this thread hasn't seen this slide before
        if not hasattr(_thread_slides, "path") or _thread_slides.path != wsi_path:
            if hasattr(_thread_slides, "handle"):
                _thread_slides.handle.close()
            _thread_slides.handle = openslide.open_slide(wsi_path)
            _thread_slides.path = wsi_path
        patch = _thread_slides.handle.read_region((x, y), read_level, (read_size, read_size))
        # Apply transform here in the worker (CPU-bound resize + normalize)
        return transform(patch.convert("RGB"))

    # Step 3: extract features — parallel I/O + batched GPU inference
    all_features = []
    with torch.no_grad():
        # Keep the executor alive across all batches so threads (and their slide
        # handles) are reused rather than re-created for every batch.
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for i in range(0, len(coords), batch_size):
                batch_coords = coords[i:i + batch_size]

                # Read all patches in this mini-batch in parallel (overlaps disk I/O)
                patches = list(executor.map(read_patch, batch_coords))

                batch = torch.stack(patches).to(device)

                # Forward pass through cTransPath (FP16 on GPU for ~2× speedup)
                with torch.autocast(device.type, enabled=(device.type == "cuda")):
                    out = model(batch)
                # cTransPath may return a tuple; take the first element
                if isinstance(out, (tuple, list)):
                    out = out[0]
                # Normalise to (B, C): timm's SwinTransformer returns a 4D spatial
                # map (B, H, W, C) when the head is Identity (pooling was inside the
                # head). Some versions return (B, num_tokens, C). Either way, average
                # over every non-batch, non-channel dimension to get (B, 768).
                if out.ndim == 4:        # (B, H, W, C) — channels-last spatial map
                    out = out.mean(dim=(1, 2))
                elif out.ndim == 3:      # (B, num_tokens, C)
                    out = out.mean(dim=1)
                all_features.append(out.cpu())

    features = torch.cat(all_features, dim=0)  # (N, 768)
    return features


# =============================================================================
# SECTION 3: DATA SPLITS
# =============================================================================

def load_label_file(label_file: str, wsi_dir: str = None):
    """
    Parse the label file and return a list of slide records.

    The label file is a tab-separated file (with quoted strings) in R format:
      - An unnamed first column with row indices
      - pid    : patient ID (multiple slides can share the same patient)
      - filename: slide file name (e.g., "10499.svs")
      - vi.label: vascular invasion label for this slide
                  VITUMOR    — tumor slide from a VI-positive patient
                  NONVITUMOR — tumor slide from a VI-negative patient
                  NONTUMOR   — non-tumor tissue (e.g., normal lung)
      - sp.label: spatial pattern label (not used here)

    Parameters
    ----------
    label_file : path to hackathon_label.txt
    wsi_dir    : if provided, only return slides whose .svs file exists in this
                 directory. Use this to restrict the dataset to a specific batch.

    Returns
    -------
    slides : list of dicts with keys: pid, filename, vi_label
    """
    # Build a set of filenames present in wsi_dir (if filtering is requested)
    available_files = None
    if wsi_dir is not None:
        available_files = set(os.listdir(wsi_dir))

    slides = []
    with open(label_file, newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)  # ['pid', 'filename', 'vi.label', 'sp.label']
        for row in reader:
            # Row format: [row_idx, pid, filename, vi_label, sp_label]
            if len(row) < 4:
                continue
            _, pid, filename, vi_label = row[0], row[1], row[2], row[3]
            # Skip slides not present in the WSI directory (when filtering)
            if available_files is not None and filename not in available_files:
                continue
            slides.append({"pid": pid, "filename": filename, "vi_label": vi_label})

    if available_files is not None:
        print(f"Filtered to {len(slides)} slides found in {wsi_dir}")
    return slides


def assign_patient_label(slide_labels: list):
    """
    Determine a patient's VI label from their set of slide labels.

    A patient is VITUMOR if any of their slides shows vascular invasion;
    otherwise NONVITUMOR. This function is called after NONTUMOR slides
    have been excluded from the input list.
    """
    if "VITUMOR" in slide_labels:
        return "VITUMOR"
    return "NONVITUMOR"


def create_patient_splits(slides: list, n_folds: int = 5, random_seed: int = 42):
    """
    Create n-fold stratified cross-validation splits at the patient level.

    Why patient-level splits?
    -------------------------
    The same patient contributes multiple slides. If we split at the slide level,
    slides from the same patient could appear in both train and test — this
    inflates performance and is a form of data leakage. Patient-level splits ensure
    that all slides from a patient are in either train or test, never both.

    Why stratified?
    ---------------
    With imbalanced classes, random splits might put most VI-positive patients in
    train and few in val. Stratification ensures each fold's train/val have the same
    class ratio as the full dataset.

    Parameters
    ----------
    slides      : list of slide dicts (from load_label_file)
    n_folds     : number of cross-validation folds (5)
    random_seed : random seed for reproducibility

    Returns
    -------
    folds : list of n_folds dicts, each with:
              "train": list of {pid, filename, vi_label} for training slides
              "test" : list of {pid, filename, vi_label} for test slides
    """
    from sklearn.model_selection import StratifiedKFold
    from collections import Counter

    # Group slides by patient
    pid_to_slides = defaultdict(list)
    for s in slides:
        pid_to_slides[s["pid"]].append(s)

    # Assign patient-level labels
    pids = sorted(pid_to_slides.keys())
    pid_labels = [assign_patient_label([s["vi_label"] for s in pid_to_slides[p]])
                  for p in pids]

    print(f"\nPatient-level label distribution:")
    label_dist = Counter(pid_labels)
    for label, count in sorted(label_dist.items()):
        print(f"  {label}: {count} patients")
    print(f"  Total: {len(pids)} patients, {len(slides)} slides\n")

    # Cap n_folds to the smallest class size (StratifiedKFold requires at least
    # n_splits samples per class)
    min_class_count = min(label_dist.values())
    if n_folds > min_class_count:
        n_folds = min_class_count
        print(f"  Note: n_folds reduced to {n_folds} (smallest class has {min_class_count} patients)\n")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    pids_arr = np.array(pids)
    labels_arr = np.array(pid_labels)

    folds = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(pids_arr, labels_arr)):
        train_pids = set(pids_arr[train_idx])
        test_pids  = set(pids_arr[test_idx])

        train_slides = [s for s in slides if s["pid"] in train_pids]
        test_slides  = [s for s in slides if s["pid"] in test_pids]

        folds.append({"train": train_slides, "test": test_slides})

        # Print fold summary
        train_labels = Counter(s["vi_label"] for s in train_slides)
        test_labels  = Counter(s["vi_label"] for s in test_slides)
        print(f"Fold {fold_idx}: "
              f"train={len(train_slides)} slides ({len(train_pids)} patients) | "
              f"test={len(test_slides)} slides ({len(test_pids)} patients)")
        print(f"  train labels: {dict(train_labels)}")
        print(f"  test  labels: {dict(test_labels)}")

    return folds


def save_splits(folds: list, splits_dir: str):
    """
    Save cross-validation splits to JSON files.

    Each file (fold_0.json ... fold_4.json) contains:
      {
        "train": [{"pid": ..., "filename": ..., "vi_label": ...}, ...],
        "test":  [{"pid": ..., "filename": ..., "vi_label": ...}, ...]
      }

    Participants can load these directly in run.py to reproduce the same splits.
    """
    os.makedirs(splits_dir, exist_ok=True)
    for i, fold in enumerate(folds):
        out_path = os.path.join(splits_dir, f"fold_{i}.json")
        with open(out_path, "w") as f:
            json.dump(fold, f, indent=2)
        print(f"Saved fold {i} → {out_path}")


# =============================================================================
# SECTION 4: MAIN PIPELINE
# =============================================================================

def run_splits_only(label_file: str, wsi_dir: str, splits_dir: str,
                    n_folds: int = 5, random_seed: int = 42):
    """Run only the data splitting step (no image processing needed)."""
    print("=" * 60)
    print("STEP: Creating cross-validation splits")
    print("=" * 60)
    slides = load_label_file(label_file, wsi_dir=wsi_dir)

    # The baseline task is binary: VITUMOR vs NONVITUMOR.
    # Exclude NONTUMOR slides from splits — they are available as
    # additional data but are not part of the binary classification target.
    tumor_slides = [s for s in slides if s["vi_label"] in ("VITUMOR", "NONVITUMOR")]
    print(f"Kept {len(tumor_slides)} VITUMOR/NONVITUMOR slides for splits "
          f"({len(slides) - len(tumor_slides)} NONTUMOR slides excluded from splits "
          f"but features are available for optional use)")

    folds = create_patient_splits(tumor_slides, n_folds=n_folds, random_seed=random_seed)
    save_splits(folds, splits_dir)
    print(f"\nDone! Splits saved to: {splits_dir}")


def run_full_pipeline(args):
    """
    Run the complete preprocessing pipeline:
      1. Create splits
      2. For each slide: segment → tile → extract features → save
    """
    import glob

    # --- Setup device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Step 1: Create splits ---
    slides = load_label_file(args.label_file, wsi_dir=args.wsi_dir)

    # The baseline task is binary: VITUMOR vs NONVITUMOR.
    # Build splits from tumor slides only; NONTUMOR features are still
    # extracted below so participants can leverage them if they wish.
    tumor_slides = [s for s in slides if s["vi_label"] in ("VITUMOR", "NONVITUMOR")]
    print(f"Kept {len(tumor_slides)} VITUMOR/NONVITUMOR slides for splits "
          f"({len(slides) - len(tumor_slides)} NONTUMOR slides excluded from splits "
          f"but their features will still be extracted for optional use)")

    folds = create_patient_splits(tumor_slides, n_folds=args.n_folds, random_seed=args.random_seed)
    save_splits(folds, args.splits_dir)

    # --- Step 2: Load cTransPath ---
    print("\n" + "=" * 60)
    print("STEP: Loading cTransPath feature extractor")
    print("=" * 60)
    model = load_ctranspath(args.ctranspath_weights, device)
    transform = get_patch_transforms()
    print("cTransPath loaded successfully.")

    # --- Step 3: Extract features for all slides ---
    print("\n" + "=" * 60)
    print("STEP: Extracting patch features from WSIs")
    print("=" * 60)
    os.makedirs(args.output_dir, exist_ok=True)

    # Build a mapping from filename to full path
    wsi_files = glob.glob(os.path.join(args.wsi_dir, "*.svs"))
    wsi_map = {Path(p).name: p for p in wsi_files}
    print(f"Found {len(wsi_map)} WSI files in {args.wsi_dir}")

    processed = 0
    skipped = 0
    for slide in slides:
        filename = slide["filename"]
        save_path = os.path.join(args.output_dir, filename.replace(".svs", ".pt"))

        # Skip if already processed
        if os.path.exists(save_path):
            processed += 1
            continue

        if filename not in wsi_map:
            print(f"  [SKIP] {filename} not found in WSI directory")
            skipped += 1
            continue

        print(f"[{processed + 1}/{len(slides)}] Processing {filename} ...")
        try:
            features = extract_features_for_slide(
                wsi_path=wsi_map[filename],
                model=model,
                transform=transform,
                device=device,
                patch_size=args.patch_size,
                step_size=args.step_size,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                tissue_thresh=args.tissue_thresh,
                read_level=args.read_level,
            )
            if features is not None:
                torch.save(features, save_path)
                print(f"  Saved features: shape={features.shape} → {save_path}")
            processed += 1
        except Exception as e:
            print(f"  [ERROR] Failed to process {filename}: {e}")
            skipped += 1

    print(f"\nDone! Processed {processed} slides, skipped {skipped}.")
    print(f"Features saved to: {args.output_dir}")


# =============================================================================
# ENTRY POINT
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocessing pipeline for VI-LUAD WSIs"
    )
    # Required for splitting
    parser.add_argument(
        "--label_file",
        type=str,
        default="/projectnb/medaihack/VI_LUAD_Project/Clinical_Data/hackathon_label.txt",
        help="Path to hackathon_label.txt",
    )
    parser.add_argument(
        "--splits_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "splits"),
        help="Directory to save cross-validation split JSON files",
    )
    parser.add_argument(
        "--n_folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducible cross-validation splits (default: 42)",
    )
    parser.add_argument(
        "--splits_only",
        action="store_true",
        help="Only generate splits, skip image processing",
    )
    # Required for full pipeline
    parser.add_argument(
        "--wsi_dir",
        type=str,
        default="/projectnb/medaihack/VI_LUAD_Project/WSI_Data/wsi",
        help="Directory containing .svs WSI files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "features"),
        help="Directory to save patch feature .pt files",
    )
    parser.add_argument(
        "--ctranspath_weights",
        type=str,
        default="/projectnb/vkolagrp/lingyixu/medaihack/ctranspath.pth",
        help="Path to cTransPath pretrained weights (.pth)",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=1024,
        help="Patch size in pixels at level 0. Use 1024 for 40× slides (resized to "
             "224×224 for cTransPath, yielding ~10× effective resolution).",
    )
    parser.add_argument(
        "--step_size",
        type=int,
        default=1024,
        help="Stride between patches in pixels (default: equal to patch_size, non-overlapping)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Number of patches per GPU forward pass through cTransPath (reduce if OOM)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of parallel threads for reading patches from disk (default: 8)",
    )
    parser.add_argument(
        "--tissue_thresh",
        type=float,
        default=0.5,
        help="Minimum tissue fraction (0–1) for a patch to be kept (default: 0.5)",
    )
    parser.add_argument(
        "--read_level",
        type=int,
        default=1,
        help="WSI pyramid level to read patches from (0=40×, 1=10×, 2=2.5×). "
             "Level 1 is ~8× faster than level 0 on NFS with the same effective "
             "magnification for cTransPath (default: 1)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.splits_only:
        run_splits_only(args.label_file, args.wsi_dir, args.splits_dir, args.n_folds, args.random_seed)
    else:
        run_full_pipeline(args)
