"""
Shared dataset for Amyloid PET Centiloid Prediction.

All team members should import from here. To add augmentations,
pass a `transform` callable that takes and returns a (1,128,128,128) tensor.
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
        csv_path: Path to CSV with npy_path and TRACER.AMY columns
                  (CENTILOIDS optional).
        tracer_map: Reuse an existing tracer->int mapping (pass train_ds.tracer_map).
        cache: If True, keep all volumes in RAM after first load (~8MB each).
        transform: Optional callable for augmentation, applied to each image tensor.
    """

    def __init__(self, csv_path: str, tracer_map: dict = None,
                 cache: bool = False, transform=None):
        df = pd.read_csv(csv_path)

        required = ["npy_path", "TRACER.AMY"]
        self.has_targets = "CENTILOIDS" in df.columns
        if self.has_targets:
            required.append("CENTILOIDS")

        df = df.dropna(subset=required)
        df = df[df["npy_path"].apply(lambda p: Path(p).exists())].reset_index(drop=True)
        print(f"Loaded {len(df)} samples from {csv_path}")

        self.df = df
        self.paths = df["npy_path"].tolist()
        self._cache = {} if cache else None
        self.transform = transform

        if self.has_targets:
            self.centiloids = torch.tensor(df["CENTILOIDS"].values, dtype=torch.float32)

        if tracer_map is None:
            unique_tracers = sorted(df["TRACER.AMY"].unique())
            self.tracer_map = {name: idx for idx, name in enumerate(unique_tracers)}
        else:
            self.tracer_map = tracer_map

        self.tracers = torch.tensor(df["TRACER.AMY"].map(self.tracer_map).values, dtype=torch.long)
        print(f"  Tracer map: {self.tracer_map}")
        if self.has_targets:
            print(f"  Centiloid range: [{self.centiloids.min():.1f}, {self.centiloids.max():.1f}]")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        if self._cache is not None and idx in self._cache:
            image = self._cache[idx]
        else:
            image = torch.from_numpy(
                np.load(self.paths[idx]).astype(np.float32)
            )  # (1, 128, 128, 128)
            if self._cache is not None:
                self._cache[idx] = image

        if self.transform is not None:
            image = self.transform(image)

        if self.has_targets:
            return image, self.centiloids[idx], self.tracers[idx]
        return image, self.tracers[idx]
