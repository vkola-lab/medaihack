"""
Shared dataset for Amyloid PET Centiloid Prediction.

Change vs previous version
───────────────────────────
__getitem__ now supports transform as either:
  • callable  — applied to every sample identically (original behaviour)
  • dict      — {tracer_id (int) : callable}, applies a different transform
                per tracer; used for tracer-aware augmentation strength
                (e.g. stronger aug for NAV n=85).
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path


class PETDataset(Dataset):
    """Load preprocessed PET .npy volumes, optionally with centiloid targets.

    Works for both training (with targets) and inference (without).
    If the CSV has a CENTILOIDS column, targets are returned; otherwise
    __getitem__ returns (image, tracer) only.

    Args:
        csv_path:   Path to CSV with npy_path and TRACER.AMY columns
                    (CENTILOIDS optional).
        tracer_map: Reuse an existing tracer->int mapping (pass train_ds.tracer_map).
        cache:      If True, keep all volumes in RAM after first load (~8 MB each).
        transform:  Optional augmentation. Either:
                      - callable (1,D,H,W) → (1,D,H,W)   applied to all samples
                      - dict {tracer_id: callable}         per-tracer augmentation
    """

    def __init__(
        self,
        csv_path:   str,
        tracer_map: dict = None,
        cache:      bool = False,
        transform        = None,
    ):
        df = pd.read_csv(csv_path)

        required = ["npy_path", "TRACER.AMY"]
        self.has_targets = "CENTILOIDS" in df.columns
        if self.has_targets:
            required.append("CENTILOIDS")

        df = df.dropna(subset=required)
        df = df[df["npy_path"].apply(lambda p: Path(p).exists())].reset_index(drop=True)
        print(f"Loaded {len(df)} samples from {csv_path}")

        self.df        = df
        self.paths     = df["npy_path"].tolist()
        self._cache    = {} if cache else None
        self.transform = transform

        if self.has_targets:
            self.centiloids = torch.tensor(
                df["CENTILOIDS"].values, dtype=torch.float32
            )

        if tracer_map is None:
            unique_tracers  = sorted(df["TRACER.AMY"].unique())
            self.tracer_map = {name: idx for idx, name in enumerate(unique_tracers)}
        else:
            self.tracer_map = tracer_map

        self.tracers = torch.tensor(
            df["TRACER.AMY"].map(self.tracer_map).values, dtype=torch.long
        )

        print(f"  Tracer map : {self.tracer_map}")
        if self.has_targets:
            print(f"  CL range   : [{self.centiloids.min():.1f},"
                  f" {self.centiloids.max():.1f}]")

        # Log augmentation mode
        if transform is None:
            print("  Augmentation: none")
        elif isinstance(transform, dict):
            print(f"  Augmentation: per-tracer dict ({len(transform)} entries)")
        else:
            print(f"  Augmentation: {type(transform).__name__}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # ── Load volume ───────────────────────────────────────────────────
        if self._cache is not None and idx in self._cache:
            image = self._cache[idx]
        else:
            image = torch.from_numpy(
                np.load(self.paths[idx]).astype(np.float32)
            )  # (1, 128, 128, 128)
            if self._cache is not None:
                self._cache[idx] = image

        # ── Augmentation ──────────────────────────────────────────────────
        if self.transform is not None:
            if isinstance(self.transform, dict):
                # Per-tracer: look up by integer tracer id
                tid = self.tracers[idx].item()
                t   = self.transform.get(tid)
                if t is not None:
                    image = t(image)
            else:
                # Uniform: same transform for all tracers
                image = self.transform(image)

        # ── Return ────────────────────────────────────────────────────────
        if self.has_targets:
            return image, self.centiloids[idx], self.tracers[idx]
        return image, self.tracers[idx]