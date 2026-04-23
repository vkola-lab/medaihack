"""
Loss functions for Amyloid PET Centiloid Prediction.

EDA motivation for CentiloidLoss:
  - Distribution is right-skewed (median ≈ 10, long upper tail)
  - Outliers exist: CL as low as -50, as high as 200+
  - HuberLoss(δ=25) is MSE-like inside ±25 CL (covers the IQR)
    and L1-like beyond — robust to the long tail without ignoring it.
  - Pearson term directly aligns training with the secondary eval metric
    and prevents a constant-prediction collapse under Huber alone.
"""

import torch
import torch.nn as nn


class CentiloidLoss(nn.Module):
    """
    Combined Huber + Pearson correlation loss.

    Loss = α · Huber(δ) + (1−α) · (1 − Pearson_r)

    Args:
        delta: Huber threshold. Default 25 ≈ IQR (P25=−1.5 → P75=47.2).
        alpha: Weight on Huber term; (1−alpha) goes to Pearson. Default 0.7.
    """

    def __init__(self, delta: float = 25.0, alpha: float = 0.7):
        super().__init__()
        self.huber = nn.HuberLoss(delta=delta, reduction="mean")
        self.alpha = alpha

    def pearson_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """1 − Pearson r  (0 = perfect correlation, 2 = perfect anti-correlation)."""
        vx = pred   - pred.mean()
        vy = target - target.mean()
        r  = (vx * vy).sum() / (vx.norm() * vy.norm() + 1e-8)
        return 1.0 - r

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.squeeze()
        return (
            self.alpha       * self.huber(pred, target)
            + (1 - self.alpha) * self.pearson_loss(pred, target)
        )


def get_criterion(name: str = "centiloid", **kwargs) -> nn.Module:
    """
    Loss factory.

    Args:
        name: "centiloid" (recommended) | "huber" | "mse" | "mae"
        **kwargs: forwarded to the constructor (e.g. delta=, alpha=).
    """
    registry = {
        "centiloid": CentiloidLoss,
        "huber":     lambda **kw: nn.HuberLoss(delta=kw.get("delta", 25.0)),
        "mse":       lambda **kw: nn.MSELoss(),
        "mae":       lambda **kw: nn.L1Loss(),
    }
    if name not in registry:
        raise ValueError(f"Unknown loss {name!r}. Choose from: {list(registry)}")
    return registry[name](**kwargs)