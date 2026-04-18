"""
Stage-3 variant: PETResNet-NoFiLM + CBAM-style spatial attention at stage-4.

Extends PETResNetNoFiLM with a single spatial attention module applied to the
stage-4 feature map (before global average pooling). The attention map is a
learnable, per-subject, continuous analog of the fixed Klunk CTX VOI used in
the traditional Centiloid pipeline.

  Kept (from petresnet_no_film):
    - TracerNorm           (per-tracer intensity γ/β — trained, not frozen)
    - 3D ResNet-18 backbone (no FiLM)
    - Regression head (no final activation, no tracer embedding concat)

  Added:
    - SpatialAttention3D on stage-4 output only (~98 params for kernel=7)

No channel attention (SE) and no attention on stages 1/2/3 — early-stage
attention over-fits on ~2k samples (see docstring on Non-goals in stage3 spec).

The forward signature accepts `return_attention=True` for visualization; by
default the flag is False so training / evaluation code is unchanged.
"""

import torch
import torch.nn as nn

from mygo_centiloid.model.petresnet_film import TracerNorm, ResBlock3D


class SpatialAttention3D(nn.Module):
    """
    CBAM-style 3D spatial attention.

    Collapses channels with a (avg, max) pair, runs a single 3D conv with
    a large receptive field (default 7³), and sigmoid-gates the input.

    Shape: (B, C, D, H, W) → (B, C, D, H, W), attn (B, 1, D, H, W)
    Params (kernel=7): 2 · 7³ = 686 weights (+0 bias).
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size,
                              padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        avg_pool = x.mean(dim=1, keepdim=True)
        max_pool = x.amax(dim=1, keepdim=True)
        pooled   = torch.cat([avg_pool, max_pool], dim=1)
        attn     = torch.sigmoid(self.conv(pooled))
        return x * attn, attn


class PETResNetAttn(nn.Module):
    """
    3D ResNet-18 + TracerNorm + Stage-4 Spatial Attention.

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
        SpatialAttention3D                  → (B, 512,  4³)  +  attn (B,1,4³)
              ↓
        Global Average Pool                 → (B, 512)
              ↓
        FC(512→256) → BN → GELU → Drop(0.4)
        FC(256→ 64) → BN → GELU → Drop(0.2)
        FC( 64→  1)   ← NO activation (centiloids can be negative)
    """

    def __init__(
        self,
        num_tracers:       int,
        dropout_high:      float = 0.4,
        dropout_low:       float = 0.2,
        mean_centiloid:    float = 0.0,
        attention_kernel:  int   = 7,
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

        self.spatial_attn = SpatialAttention3D(kernel_size=attention_kernel)

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

    def forward(
        self,
        x: torch.Tensor,
        tracer_idx: torch.Tensor,
        return_attention: bool = False,
    ):
        x = self.tracer_norm(x, tracer_idx)

        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x, attn = self.spatial_attn(x)

        x = self.gap(x).flatten(1)
        y = self.head(x).squeeze(1)

        if return_attention:
            return y, attn
        return y

    def summary(self, input_size=(1, 1, 128, 128, 128), depth: int = 4) -> None:
        bar = "=" * 88
        print(bar)
        print(f"{'PETResNet-Attn (stage-4 spatial attention) Architecture':^88}")
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
            attn  = sum(p.numel() for p in self.spatial_attn.parameters())
            print("-" * 88)
            print(f"Total parameters:     {total:,}")
            print(f"Trainable parameters: {train:,}")
            print(f"Spatial attn params:  {attn:,}")
            print(f"Num tracers:          {num_tracers}")
            print("(Install torchinfo for per-layer shapes: pip install torchinfo)")

        print(bar + "\n")
