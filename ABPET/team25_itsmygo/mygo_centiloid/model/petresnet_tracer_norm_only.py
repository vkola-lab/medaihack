"""
Ablation: PETResNet with TracerNorm as the ONLY tracer-conditioning site.

This is the report's **TracerNorm-only** variant (§3.5, Table 2; MAE 9.03 CL
on the internal 500-subject validation split, §4.3 Table 4). FiLM blocks and
the tracer-embedding GAP concat are both removed; the tracer identity reaches
the network exclusively through TracerNorm's per-tracer (γ_t, β_t) applied to
the raw input volume.

  Kept:
    - TracerNorm           (per-tracer input-space γ/β, 8 learned scalars)
    - 3D ResNet-18 backbone
    - Regression head (no final activation — Centiloids can be negative)

  Removed:
    - Shared tracer embedding (`tracer_emb`) — no downstream embedding at all
    - FiLM blocks at every ResNet stage
    - Tracer-embedding concat into the regression head (feat_dim: 544 → 512)

The forward signature still accepts `tracer_idx` so training / evaluation
code that targets PETResNet can drop in this ablation unchanged — the id is
routed to TracerNorm only.
"""

import torch
import torch.nn as nn

from mygo_centiloid.model.petresnet import TracerNorm, ResBlock3D


class PETResNetTracerNormOnly(nn.Module):
    """
    3D ResNet-18 + TracerNorm, with FiLM and tracer-embedding concat removed.

    Forward pass:
        TracerNorm                           per-tracer intensity fix
              ↓
        Stem: Conv7³(s=2) + MaxPool(s=2)    (B,  1,128³) → (B, 64,32³)
              ↓
        Stage 1: ResBlock × 2  (stride=1)   → (B,  64, 32³)
        Stage 2: ResBlock × 2  (stride=2)   → (B, 128, 16³)
        Stage 3: ResBlock × 2  (stride=2)   → (B, 256,  8³)
        Stage 4: ResBlock × 2  (stride=2)   → (B, 512,  4³)
              ↓
        Global Average Pool                 → (B, 512)
              ↓
        FC(512→256) → BN → GELU → Drop(0.4)
        FC(256→ 64) → BN → GELU → Drop(0.2)
        FC( 64→  1)   ← NO activation (centiloids can be negative)
    """

    def __init__(
        self,
        num_tracers:    int,
        dropout_high:   float = 0.4,
        dropout_low:    float = 0.2,
        mean_centiloid: float = 0.0,
    ):
        super().__init__()

        self.tracer_norm = TracerNorm(num_tracers)

        self.stem = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )

        self.stage1 = self._make_stage(64,   64, n=2, stride=1)
        self.stage2 = self._make_stage(64,  128, n=2, stride=2)
        self.stage3 = self._make_stage(128, 256, n=2, stride=2)
        self.stage4 = self._make_stage(256, 512, n=2, stride=2)

        self.gap = nn.AdaptiveAvgPool3d(1)

        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout_high),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout_low),
            nn.Linear(64, 1),
        )
        nn.init.constant_(self.head[-1].bias, mean_centiloid)

    @staticmethod
    def _make_stage(in_ch, out_ch, n, stride):
        return nn.Sequential(
            ResBlock3D(in_ch, out_ch, stride=stride),
            *[ResBlock3D(out_ch, out_ch) for _ in range(1, n)],
        )

    def forward(self, x: torch.Tensor, tracer_idx: torch.Tensor) -> torch.Tensor:
        x = self.tracer_norm(x, tracer_idx)

        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.gap(x).flatten(1)
        return self.head(x).squeeze(1)

    def summary(self, input_size=(1, 1, 128, 128, 128), depth: int = 4) -> None:
        bar = "=" * 88
        print(bar)
        print(f"{'PETResNet-TracerNormOnly (ablation) Architecture':^88}")
        print(bar)

        device      = next(self.parameters()).device
        num_tracers = self.tracer_norm.scale.num_embeddings
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
