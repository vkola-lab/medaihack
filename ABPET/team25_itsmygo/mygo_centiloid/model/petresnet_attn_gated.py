"""
Stage-3 variant v2: residual spatial attention with zero-init.

Fixes two pathologies observed in petresnet_attn:
  1. CBAM gating (x * sigmoid) can suppress signal to near-zero at any
     location, which collapses backbone features early and hurts Pearson r
     at epoch 1 (0.25 vs baseline 0.78).
  2. Default conv init randomly biases attention away from uniform on
     step 0, disrupting features before the optimizer has any gradient
     signal about *which* regions matter.

Changes vs petresnet_attn.py (everything else identical):
  • SpatialAttention3DGated.forward returns  x * (1 + attn), attn
        ─ attn ∈ [0,1] → effective multiplier ∈ [1,2] (enhancement only).
  • SpatialAttention3DGated.__init__ zero-inits conv weight
        ─ at step 0: sigmoid(0) = 0.5 everywhere → x_out = 1.5 · x_in
          (a uniform scalar, absorbed by the downstream BN).

Effect: the attention module starts as effective identity (pre-BN:
1.5x scalar; post-BN: identity) and gradient descent decides when and
how to deviate.

Everything else (TracerNorm, 3D ResNet-18 stages, no FiLM, no GAP
embedding concat, regression head) is identical to petresnet_attn.py
and petresnet_no_film.py.
"""

import torch
import torch.nn as nn

from mygo_centiloid.model.petresnet_film import TracerNorm, ResBlock3D


class SpatialAttention3DGated(nn.Module):
    """
    Residual CBAM-style 3D spatial attention with zero-init.

    Shape: (B, C, D, H, W) → (B, C, D, H, W), attn (B, 1, D, H, W)

    Deviations from SpatialAttention3D:
      • forward returns x * (1 + attn) instead of x * attn
      • conv weights initialized to zero (→ sigmoid=0.5 everywhere at step 0)
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size,
                              padding=padding, bias=False)
        nn.init.zeros_(self.conv.weight)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        avg_pool = x.mean(dim=1, keepdim=True)
        max_pool = x.amax(dim=1, keepdim=True)
        pooled   = torch.cat([avg_pool, max_pool], dim=1)
        attn     = torch.sigmoid(self.conv(pooled))
        return x * (1.0 + attn), attn


class PETResNetAttnGated(nn.Module):
    """
    3D ResNet-18 + TracerNorm + Stage-4 Residual Spatial Attention.

    Forward pass matches PETResNetAttn exactly, except the spatial attention
    is residual/enhancement-only and starts as effective identity. See module
    docstring for rationale.
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

        self.spatial_attn = SpatialAttention3DGated(kernel_size=attention_kernel)

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
        print(f"{'PETResNet-Attn-Gated (residual + zero-init) Architecture':^88}")
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


# ─────────────────────────────────────────────────────────────────────────────
# Init sanity check
#   Run:  python -m mygo_centiloid.model.petresnet_attn_gated
#   Asserts that at step 0 the attention module is effective identity.
# ─────────────────────────────────────────────────────────────────────────────

def _sanity_check_init() -> None:
    torch.manual_seed(0)
    attn = SpatialAttention3DGated(kernel_size=7).eval()

    # 1. Conv weights exactly zero
    wsum = attn.conv.weight.abs().sum().item()
    assert wsum == 0.0, f"conv.weight.sum()={wsum} (expected 0)"
    print(f"[1/3] conv.weight sum of |·| = {wsum}  ✓")

    # 2. sigmoid(conv(pooled)) ≈ 0.5 everywhere
    x = torch.randn(2, 512, 4, 4, 4)
    x_out, a = attn(x)
    print(f"[2/3] raw attn: min={a.min():.6f}  max={a.max():.6f}  "
          f"mean={a.mean():.6f}  std={a.std():.6f}")
    assert torch.allclose(a, torch.full_like(a, 0.5), atol=1e-6), \
        "attention not uniformly 0.5 at init"

    # 3. Output ≈ 1.5 · input at init
    expected = 1.5 * x
    max_err  = (x_out - expected).abs().max().item()
    print(f"[3/3] max |x_out - 1.5·x_in| = {max_err:.3e}  ✓")
    assert max_err < 1e-5, f"x_out != 1.5·x_in (max err {max_err})"

    print("\nAll three init invariants hold — attention is effective identity "
          "at step 0.")


if __name__ == "__main__":
    _sanity_check_init()
