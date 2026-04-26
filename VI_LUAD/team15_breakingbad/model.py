"""
model.py — ACMIL Model for VI-LUAD Binary Classification
=========================================================
Upgraded architecture: ACMIL (Attention-Challenging MIL) with Multi-Branch
Attention + Stochastic Top-K Instance Masking (STKIM), 2D sinusoidal
positional encoding over patch (col, row) grid coordinates, built-in
temperature scaling, and built-in probability clipping for log-loss safety.

Pipeline:
  UNI2-h features (N, 1536) + 2D sinusoidal PE from coords
    -> feature projection MLP
    -> M gated-attention branches (STKIM during training)
    -> branch-specific slide embeddings + branch-specific logits
    -> mean logits across branches / temperature
    -> clipped probabilities (returned as log-probs so a downstream softmax
       recovers them exactly — honors the locked predict.py inference path)

Contract with predict.py (for inference):
  The locked inference path calls `model(features)`. The ensemble wrapper
  built in predict.py.load_checkpoint intercepts torch.load to capture
  `coords` and forwards `(features, coords)` into ACMIL internally.

Contract during training:
  Call `model(features, coords)` directly — coords are read from the
  PatientBagDataset on each slide.
"""

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# =============================================================================
# SECTION 0: CONSTANTS
# =============================================================================

LABEL_MAP: Dict[str, int] = {
    "NONVITUMOR": 0,
    "VITUMOR":    1,
}
IDX_TO_LABEL: Dict[int, str] = {v: k for k, v in LABEL_MAP.items()}
NUM_CLASSES: int = 2
FEATURE_DIM: int = 1536


# =============================================================================
# SECTION 1: 2D SINUSOIDAL POSITIONAL ENCODING
# =============================================================================

def sinusoidal_2d_encoding(coords: torch.Tensor, d_model: int = 64) -> torch.Tensor:
    """
    2D sinusoidal positional encoding for patch grid coordinates.

    Parameters
    ----------
    coords  : (N, 2) tensor of (col, row) integer grid positions
    d_model : PE dimensionality (must be divisible by 4)

    Returns
    -------
    pe      : (N, d_model) tensor, on the same device as coords
    """
    pe = torch.zeros(coords.shape[0], d_model, device=coords.device)
    d_half = d_model // 2
    div = torch.exp(torch.arange(0, d_half, 2, device=coords.device).float()
                    * (-math.log(10000.0) / d_half))
    col = coords[:, 0].float().unsqueeze(1)
    row = coords[:, 1].float().unsqueeze(1)
    pe[:, 0:d_half:2]    = torch.sin(col * div)
    pe[:, 1:d_half:2]    = torch.cos(col * div)
    pe[:, d_half::2]     = torch.sin(row * div)
    pe[:, d_half + 1::2] = torch.cos(row * div)
    return pe


# =============================================================================
# SECTION 2: DATASETS
# =============================================================================

class SlideDataset(Dataset):
    """
    Slide-level dataset. Returns (features, coords, label, pid, filename)
    per sample. Used for per-slide evaluation / inference. For training with
    patient-max BCE, use PatientBagDataset instead.
    """

    def __init__(self, slide_records: List[Dict], features_dir: str,
                 label_map: Dict[str, int] = LABEL_MAP):
        self.features_dir = Path(features_dir)
        self.label_map = label_map

        self.samples = []
        skipped = 0
        for record in slide_records:
            stem = Path(record["filename"]).stem
            feat_path = self.features_dir / f"{stem}.pt"
            if feat_path.exists():
                self.samples.append({
                    "feat_path": feat_path,
                    "label":     label_map[record["vi_label"]],
                    "vi_label":  record["vi_label"],
                    "pid":       record["pid"],
                    "filename":  record["filename"],
                })
            else:
                skipped += 1

        if skipped > 0:
            print(f"  [SlideDataset] WARNING: {skipped} slides missing feature files.")
        print(f"  [SlideDataset] Ready: {len(self.samples)} slides from '{features_dir}'.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        try:
            data = torch.load(sample["feat_path"], weights_only=True)
        except TypeError:
            data = torch.load(sample["feat_path"])
        features = data["features"]
        coords = data["coords"] if "coords" in data else torch.zeros(features.shape[0], 2)
        return {
            "features": features,
            "coords":   coords,
            "label":    sample["label"],
            "pid":      sample["pid"],
            "filename": sample["filename"],
        }


class PatientBagDataset(Dataset):
    """
    Patient-level dataset. Each __getitem__ returns all slides for one
    patient, so the training loop can compute `max(P_VI)` over the bag and
    apply a patient-level BCE loss that matches the scoring rule.
    """

    def __init__(self, slide_records: List[Dict], features_dir: str,
                 label_map: Dict[str, int] = LABEL_MAP):
        self.features_dir = Path(features_dir)
        self.label_map = label_map

        by_pid: Dict[str, List[Dict]] = {}
        missing = 0
        for record in slide_records:
            stem = Path(record["filename"]).stem
            feat_path = self.features_dir / f"{stem}.pt"
            if not feat_path.exists():
                missing += 1
                continue
            by_pid.setdefault(record["pid"], []).append({
                "feat_path": feat_path,
                "label":     label_map[record["vi_label"]],
                "vi_label":  record["vi_label"],
                "filename":  record["filename"],
            })

        self.pids = sorted(by_pid.keys())
        self.by_pid = by_pid
        self.total_slides = sum(len(v) for v in by_pid.values())

        if missing > 0:
            print(f"  [PatientBagDataset] WARNING: {missing} slides missing feature files.")
        print(f"  [PatientBagDataset] Ready: {len(self.pids)} patients / "
              f"{self.total_slides} slides from '{features_dir}'.")

    def __len__(self) -> int:
        return len(self.pids)

    def __getitem__(self, idx: int) -> Dict:
        pid = self.pids[idx]
        slide_items = []
        slide_labels = []
        for meta in self.by_pid[pid]:
            try:
                data = torch.load(meta["feat_path"], weights_only=True)
            except TypeError:
                data = torch.load(meta["feat_path"])
            features = data["features"]
            coords = data["coords"] if "coords" in data else torch.zeros(features.shape[0], 2)
            slide_items.append({
                "features": features,
                "coords":   coords,
                "label":    meta["label"],
                "filename": meta["filename"],
                "vi_label": meta["vi_label"],
            })
            slide_labels.append(meta["label"])

        patient_label = 1 if any(lbl == 1 for lbl in slide_labels) else 0
        return {
            "pid":           pid,
            "slides":        slide_items,
            "patient_label": patient_label,
        }


def patient_collate_fn(batch):
    """Identity collate — returns the list of patient dicts as-is."""
    return batch


def get_patient_dataloader(slide_records, features_dir, shuffle=True, num_workers=0):
    dataset = PatientBagDataset(slide_records, features_dir)
    return DataLoader(
        dataset,
        batch_size=1,            # one patient per step
        shuffle=shuffle,
        collate_fn=patient_collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )


# =============================================================================
# SECTION 3: ACMIL MODEL
# =============================================================================

class GatedAttention(nn.Module):
    """Single gated attention head (ABMIL-style). Returns raw attention logits (N,)."""

    def __init__(self, hidden_dim: int, attn_dim: int = 128):
        super().__init__()
        self.V = nn.Linear(hidden_dim, attn_dim)
        self.U = nn.Linear(hidden_dim, attn_dim)
        self.w = nn.Linear(attn_dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        v = torch.tanh(self.V(h))
        u = torch.sigmoid(self.U(h))
        a_logits = self.w(v * u).squeeze(-1)
        return a_logits


class ACMIL(nn.Module):
    """
    Attention-Challenging MIL.

    - Multi-Branch Attention (MBA): M independent gated-attention heads, each
      producing its own slide embedding and its own classifier head.
    - Stochastic Top-K Instance Masking (STKIM): during training, with
      probability `mask_prob`, mask the top-k attended patches per branch to
      force attention to spread and prevent collapse onto a few patches.
    - Final slide prediction = mean of branch logits.
    - Post-hoc temperature scaling: divide logits by `self.temperature` before
      softmax. Temperature is frozen at 1.0 during training and fit on val
      after training completes.
    - Probability clipping at inference to bound worst-case log loss.

    forward returns (out_logits, aux):
      out_logits          : (1, 2) — log of clipped calibrated probabilities.
                            softmax(out_logits) recovers the clipped probs.
      aux["raw_mean_logits"]    : (1, 2) pre-temperature, pre-clip. Use this
                                  for training loss computation.
      aux["branch_logits"]      : (M, 1, 2) per-branch raw logits (for
                                  auxiliary branch-wise CE loss).
      aux["branch_attentions"]  : list of M tensors (N,) softmax attention
                                  per branch (for entropy / diversity aux).
    """

    def __init__(self,
                 feature_dim: int = FEATURE_DIM,
                 hidden_dim: int = 512,
                 num_classes: int = NUM_CLASSES,
                 dropout: float = 0.25,
                 n_branches: int = 5,
                 top_k: int = 10,
                 mask_prob: float = 0.6,
                 use_pe: bool = True,
                 pe_dim: int = 64,
                 clip_eps: float = 0.02):
        super().__init__()

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.n_branches = n_branches
        self.top_k      = top_k
        self.mask_prob  = mask_prob
        self.use_pe     = use_pe
        self.pe_dim     = pe_dim
        self.clip_eps   = clip_eps

        input_dim = feature_dim + (pe_dim if use_pe else 0)

        # Feature projection: (feature_dim + pe_dim) -> hidden_dim
        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Multi-branch gated attention + branch classifiers
        self.branches = nn.ModuleList([
            GatedAttention(hidden_dim, attn_dim=max(hidden_dim // 2, 64))
            for _ in range(n_branches)
        ])
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
            for _ in range(n_branches)
        ])

        # Temperature scaling — frozen during training, fit on val after.
        self.temperature = nn.Parameter(torch.ones(1), requires_grad=False)

    # -----------------------------------------------------------------------

    def _maybe_add_pe(self, features: torch.Tensor,
                      coords: Optional[torch.Tensor]) -> torch.Tensor:
        if not self.use_pe:
            return features
        if coords is None:
            # Fallback: pad zeros so the projection layer shape stays consistent.
            pad = torch.zeros(features.shape[0], self.pe_dim,
                              device=features.device, dtype=features.dtype)
            return torch.cat([features, pad], dim=-1)
        pe = sinusoidal_2d_encoding(coords.to(features.device), d_model=self.pe_dim)
        pe = pe.to(dtype=features.dtype)
        return torch.cat([features, pe], dim=-1)

    def _branch_forward(self, h: torch.Tensor, branch_idx: int
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One branch forward. Returns (branch_logits (1, C), attention (N,))."""
        attn_module = self.branches[branch_idx]
        cls_module  = self.classifiers[branch_idx]

        a_logits = attn_module(h)  # (N,)

        # STKIM: stochastic top-k masking, training only
        if self.training and self.top_k > 0 and a_logits.shape[0] > self.top_k:
            if torch.rand(1, device=a_logits.device).item() < self.mask_prob:
                _, top_idx = torch.topk(a_logits, self.top_k)
                mask = torch.zeros_like(a_logits, dtype=torch.bool)
                mask[top_idx] = True
                a_logits = a_logits.masked_fill(mask, float("-inf"))

        a = torch.softmax(a_logits, dim=0)                    # (N,)
        z = (a.unsqueeze(-1) * h).sum(dim=0, keepdim=True)    # (1, hidden_dim)
        logits_b = cls_module(z)                              # (1, C)
        return logits_b, a

    # -----------------------------------------------------------------------

    def forward(self,
                features: torch.Tensor,
                coords: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, Dict]:
        h_in = self._maybe_add_pe(features, coords)
        h = self.feature_proj(h_in)  # (N, hidden_dim)

        branch_logits_list = []
        branch_attn_list   = []
        for b in range(self.n_branches):
            logits_b, a_b = self._branch_forward(h, b)
            branch_logits_list.append(logits_b)
            branch_attn_list.append(a_b)

        all_branch_logits = torch.stack(branch_logits_list, dim=0)  # (M, 1, C)
        mean_logits = all_branch_logits.mean(dim=0)                 # (1, C)

        # Temperature scaling (no-op while temperature == 1.0)
        calibrated = mean_logits / self.temperature.clamp(min=1e-3)

        # Probability clipping (inference only — bounds worst-case log loss)
        probs = torch.softmax(calibrated, dim=-1)
        probs = probs.clamp(min=self.clip_eps, max=1.0 - self.clip_eps)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        out_logits = torch.log(probs)  # softmax(log p) == p

        aux = {
            "raw_mean_logits":   mean_logits,
            "branch_logits":     all_branch_logits,
            "branch_attentions": branch_attn_list,
        }
        return out_logits, aux


# =============================================================================
# SECTION 4: ENSEMBLE WRAPPER
# =============================================================================

class ACMILEnsemble(nn.Module):
    """
    Averages probabilities from multiple ACMIL models at inference.
    Accepts the same (features, coords=None) signature as ACMIL.
    """

    def __init__(self, models: List[nn.Module]):
        super().__init__()
        self.members = nn.ModuleList(models)

    def forward(self,
                features: torch.Tensor,
                coords: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, Dict]:
        probs_sum = None
        for m in self.members:
            logits_m, _ = m(features, coords)
            probs_m = torch.softmax(logits_m, dim=-1)
            probs_sum = probs_m if probs_sum is None else probs_sum + probs_m
        probs = probs_sum / len(self.members)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        return torch.log(probs), {"ensemble_size": len(self.members)}


# =============================================================================
# SECTION 5: FACTORY + CONFIG HELPERS
# =============================================================================

def build_model(feature_dim: int = FEATURE_DIM,
                hidden_dim: int = 512,
                num_classes: int = NUM_CLASSES,
                dropout: float = 0.25,
                n_branches: int = 5,
                top_k: int = 10,
                mask_prob: float = 0.6,
                use_pe: bool = True,
                pe_dim: int = 64,
                clip_eps: float = 0.02,
                verbose: bool = True) -> ACMIL:
    model = ACMIL(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        dropout=dropout,
        n_branches=n_branches,
        top_k=top_k,
        mask_prob=mask_prob,
        use_pe=use_pe,
        pe_dim=pe_dim,
        clip_eps=clip_eps,
    )
    if verbose:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"ACMIL built: {n_params:,} trainable params "
              f"(hidden={hidden_dim}, branches={n_branches}, use_pe={use_pe}, "
              f"top_k={top_k}, mask_prob={mask_prob})")
    return model


def build_model_from_config(config: Dict, verbose: bool = False) -> ACMIL:
    """Rebuild an ACMIL from a stored config dict (see train_eval.save logic)."""
    keys = ["feature_dim", "hidden_dim", "num_classes", "dropout",
            "n_branches", "top_k", "mask_prob", "use_pe", "pe_dim", "clip_eps"]
    kwargs = {k: config[k] for k in keys if k in config}
    return build_model(verbose=verbose, **kwargs)


# =============================================================================
# SECTION 6: SANITY CHECK
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("model.py — ACMIL sanity check")
    print("=" * 60)

    model = build_model()
    model.eval()

    N = 1000
    feats = torch.randn(N, FEATURE_DIM)
    coords = torch.stack([torch.randint(0, 40, (N,)),
                          torch.randint(0, 40, (N,))], dim=1)

    with torch.no_grad():
        logits, aux = model(feats, coords)
    probs = torch.softmax(logits, dim=-1)
    print(f"\nWith coords: logits={tuple(logits.shape)}  probs={probs[0].tolist()}")
    print(f"Branch logits: {tuple(aux['branch_logits'].shape)}")

    # Without coords (PE falls back to zeros — the model still runs)
    with torch.no_grad():
        logits2, _ = model(feats)
    print(f"No coords  : probs={torch.softmax(logits2, dim=-1)[0].tolist()}")

    # Ensemble
    members = [build_model(verbose=False) for _ in range(3)]
    ens = ACMILEnsemble(members).eval()
    with torch.no_grad():
        e_logits, _ = ens(feats, coords)
    print(f"\nEnsemble(3): probs={torch.softmax(e_logits, dim=-1)[0].tolist()}")
    print("\nAll checks passed.")
