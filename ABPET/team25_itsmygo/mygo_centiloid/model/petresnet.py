"""
Improved Model Architecture for Amyloid PET Centiloid Prediction.

EDA → Design mapping:
  FBP brighter than FBB/PIB/NAV  → TracerNorm (per-tracer learned scale/shift)
  NAV most different (KS=0.240)  → FiLM conditioning at every ResNet stage
  PIB dark at high CL            → TracerNorm fixes intensity before encoding
  Negative CL values exist (-50) → No output activation (linear head)
  Small NAV set (n=85)           → Higher dropout (0.4) in head
  Simple CNN baseline            → 3D ResNet-18 for deeper feature extraction
"""

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Building Blocks
# ─────────────────────────────────────────────────────────────────────────────

class TracerNorm(nn.Module):
    """
    Learned per-tracer intensity normalization.

    EDA Finds: FBP images are visually much brighter than FBB / NAV / PIB.
    Applying a per-tracer (γ, β) normalizes each tracer's intensity
    distribution before any shared feature extraction.

    Initialized to identity (γ=1, β=0) so training starts from the
    original signal without any destructive rescaling.

    Shape: (B, 1, D, H, W) → (B, 1, D, H, W)
    """

    def __init__(self, n_tracers: int):
        super().__init__()
        self.scale = nn.Embedding(n_tracers, 1)   # γ per tracer
        self.shift = nn.Embedding(n_tracers, 1)   # β per tracer
        nn.init.ones_(self.scale.weight)
        nn.init.zeros_(self.shift.weight)

    def forward(self, x: torch.Tensor, tracer_id: torch.Tensor) -> torch.Tensor:
        γ = self.scale(tracer_id).view(-1, 1, 1, 1, 1)
        β = self.shift(tracer_id).view(-1, 1, 1, 1, 1)
        return γ * x + β


class FiLMBlock(nn.Module):
    """
    Feature-wise Linear Modulation conditioned on tracer embedding.

    EDA: NAV has the highest KS distance (0.240) from other tracers.
    FiLM allows each tracer to independently rescale & shift every
    feature map channel, going beyond a single embedding lookup.

    Initialized to identity so the residual backbone is stable at
    the start of training.

    Shape: (B, C, D, H, W) → (B, C, D, H, W)
    """

    def __init__(self, tracer_dim: int, feat_dim: int):
        super().__init__()
        self.gamma = nn.Linear(tracer_dim, feat_dim)
        self.beta  = nn.Linear(tracer_dim, feat_dim)
        # Identity initialization
        nn.init.ones_(self.gamma.weight);  nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight);  nn.init.zeros_(self.beta.bias)

    def forward(self, x: torch.Tensor, tracer_emb: torch.Tensor) -> torch.Tensor:
        B, C = x.shape[:2]
        γ = self.gamma(tracer_emb).view(B, C, 1, 1, 1)
        β = self.beta(tracer_emb).view(B, C, 1, 1, 1)
        return γ * x + β


class ResBlock3D(nn.Module):
    """
    Standard 3D residual block with optional stride-based downsampling.

    Conv3d → BN → ReLU → Conv3d → BN ──┐
         ↑──────── skip (1×1 conv) ────┘ → ReLU
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch,  out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, stride=1,      padding=1, bias=False)
        self.bn2   = nn.BatchNorm3d(out_ch)
        self.relu  = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm3d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


# ─────────────────────────────────────────────────────────────────────────────
# Full Model
# ─────────────────────────────────────────────────────────────────────────────

class PETResNet(nn.Module):
    """
    3D ResNet-18 backbone with TracerNorm + FiLM conditioning.

    Full forward pass:
        TracerNorm                           fixes intensity per tracer
              ↓
        Stem: Conv7³(s=2) + MaxPool(s=2)    (B,  1,128,128,128) → (B, 64,32,32,32)
              ↓
        Stage 1: ResBlock × 2  (stride=1)   → (B,  64, 32, 32, 32)  + FiLM
        Stage 2: ResBlock × 2  (stride=2)   → (B, 128, 16, 16, 16)  + FiLM
        Stage 3: ResBlock × 2  (stride=2)   → (B, 256,  8,  8,  8)  + FiLM
        Stage 4: ResBlock × 2  (stride=2)   → (B, 512,  4,  4,  4)  + FiLM
              ↓
        Global Average Pool                 → (B, 512)
              ↓
        Concat tracer_emb                   → (B, 544)
              ↓
        FC(544→256) → BN → GELU → Drop(0.4)
        FC(256→ 64) → BN → GELU → Drop(0.2)
        FC( 64→  1)   ← NO activation (centiloids can be negative)

    Args:
        num_tracers:    Number of distinct PET tracers.
        emb_dim:        Tracer embedding size. Default 32.
        dropout_high:   Dropout rate for first FC layer. Default 0.4.
        dropout_low:    Dropout rate for second FC layer. Default 0.2.
        mean_centiloid: Dataset mean used to initialize output bias.
    """

    def __init__(
        self,
        num_tracers:    int,
        emb_dim:        int   = 32,
        dropout_high:   float = 0.4,
        dropout_low:    float = 0.2,
        mean_centiloid: float = 0.0,
    ):
        super().__init__()

        # 1. Per-tracer intensity normalization
        self.tracer_norm = TracerNorm(num_tracers)

        # 2. Shared tracer embedding (used by FiLM blocks AND regression head)
        self.tracer_emb = nn.Embedding(num_tracers, emb_dim)

        # 3. Stem: (B,1,128,128,128) → (B,64,32,32,32)
        self.stem = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )

        # 4. ResNet stages
        self.stage1 = self._make_stage(64,   64, n=2, stride=1)
        self.stage2 = self._make_stage(64,  128, n=2, stride=2)
        self.stage3 = self._make_stage(128, 256, n=2, stride=2)
        self.stage4 = self._make_stage(256, 512, n=2, stride=2)

        # 5. FiLM blocks — one per stage
        self.film1 = FiLMBlock(emb_dim,  64)
        self.film2 = FiLMBlock(emb_dim, 128)
        self.film3 = FiLMBlock(emb_dim, 256)
        self.film4 = FiLMBlock(emb_dim, 512)

        # 6. Global Average Pool
        self.gap = nn.AdaptiveAvgPool3d(1)

        # 7. Regression head  (no final activation — CL can be negative)
        feat_dim = 512 + emb_dim  # 544
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout_high),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout_low),
            nn.Linear(64, 1),   # ← linear output
        )
        nn.init.constant_(self.head[-1].bias, mean_centiloid)

    @staticmethod
    def _make_stage(in_ch, out_ch, n, stride):
        return nn.Sequential(
            ResBlock3D(in_ch, out_ch, stride=stride),
            *[ResBlock3D(out_ch, out_ch) for _ in range(1, n)],
        )

    def forward(self, x: torch.Tensor, tracer_idx: torch.Tensor) -> torch.Tensor:
        # Tracer-aware intensity normalisation
        x = self.tracer_norm(x, tracer_idx)

        # Tracer embedding (shared throughout)
        t = self.tracer_emb(tracer_idx)          # (B, emb_dim)

        # Encode
        x = self.stem(x)                         # (B,  64, 32, 32, 32)

        x = self.film1(self.stage1(x), t)        # (B,  64, 32, 32, 32)
        x = self.film2(self.stage2(x), t)        # (B, 128, 16, 16, 16)
        x = self.film3(self.stage3(x), t)        # (B, 256,  8,  8,  8)
        x = self.film4(self.stage4(x), t)        # (B, 512,  4,  4,  4)

        x = self.gap(x).flatten(1)               # (B, 512)
        x = torch.cat([x, t], dim=1)             # (B, 544)
        return self.head(x).squeeze(1)           # (B,)

    def summary(self, input_size=(1, 1, 128, 128, 128), depth: int = 4) -> None:
        """
        Print a detailed architecture summary (per-layer shapes + param counts).

        Uses `torchinfo` when available for rich output; falls back to the
        built-in `print(self)` plus parameter totals otherwise.
        """
        bar = "=" * 88
        print(bar)
        print(f"{'mygo_centiloid Architecture':^88}")
        print(bar)

        device      = next(self.parameters()).device
        num_tracers = self.tracer_emb.num_embeddings
        dummy_x     = torch.zeros(*input_size, device=device)
        dummy_id    = torch.zeros(input_size[0], dtype=torch.long, device=device)

        try:
            from torchinfo import summary as _tsummary
            _tsummary(
                self,
                input_data=(dummy_x, dummy_id),
                depth=depth,
                col_names=("input_size", "output_size", "num_params"),
                row_settings=("depth", "var_names"),
            )
        except ImportError:
            print(self)
            total = sum(p.numel() for p in self.parameters())
            train = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print("-" * 88)
            print(f"Total parameters:     {total:,}")
            print(f"Trainable parameters: {train:,}")
            print(f"Num tracers:          {num_tracers}")
            print("(Install torchinfo for per-layer shapes: pip install torchinfo)")

        print(bar + "\n")


# Backward-compatible alias — predict.py continues to work unchanged
BaselineCNN = PETResNet