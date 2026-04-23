"""
3D augmentation pipeline for Amyloid PET volumes.

EDA motivation
──────────────
  - Brain is approximately bilaterally symmetric → safe LR flip
  - PET intensity differs across tracers (FBP bright, PIB dark) →
    gamma + bias jitter improves robustness
  - Acquisition noise varies by scanner / protocol → small Gaussian noise
  - Subjects are roughly co-registered (RAS, 2 mm iso, 128³) →
    only small affine perturbations are physically plausible

  NAV (n=85) and PIB are under-represented → use STRONG strength;
  FBP and FBB use STANDARD strength.

All transforms operate on a single tensor of shape (1, D, H, W),
in [0, 1] float32, and return the same shape / dtype / range.
"""

import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Individual transforms
# ─────────────────────────────────────────────────────────────────────────────

class RandFlipLR:
    """Flip along the left-right axis (last spatial dim) with probability p."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.p:
            return torch.flip(x, dims=[-1])
        return x


class RandGamma:
    """Random gamma correction in [1−r, 1+r] (multiplicative on [0,1])."""

    def __init__(self, r: float = 0.1, p: float = 0.5):
        self.r = r
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() >= self.p:
            return x
        gamma = 1.0 + (torch.rand(1).item() * 2 - 1) * self.r
        return x.clamp(0, 1).pow(gamma)


class RandBiasShift:
    """Add a uniform bias in [−s, s] then re-clamp to [0, 1]."""

    def __init__(self, s: float = 0.05, p: float = 0.5):
        self.s = s
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() >= self.p:
            return x
        b = (torch.rand(1).item() * 2 - 1) * self.s
        return (x + b).clamp(0, 1)


class RandGaussianNoise:
    """Additive Gaussian noise with σ ∈ [0, sigma_max], then clamp."""

    def __init__(self, sigma_max: float = 0.02, p: float = 0.5):
        self.sigma_max = sigma_max
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() >= self.p:
            return x
        sigma = torch.rand(1).item() * self.sigma_max
        return (x + torch.randn_like(x) * sigma).clamp(0, 1)


class RandAffine3D:
    """
    Small 3D affine: rotation (deg) + translation (fraction of FOV).

    Implemented with torch.affine_grid + grid_sample so it runs on the
    same device as the input tensor (CPU during DataLoader workers).
    """

    def __init__(
        self,
        rot_deg: float = 5.0,
        trans_frac: float = 0.05,
        p: float = 0.5,
    ):
        self.rot_deg    = rot_deg
        self.trans_frac = trans_frac
        self.p          = p

    @staticmethod
    def _rot_matrix(rx: float, ry: float, rz: float) -> torch.Tensor:
        cx, sx = torch.cos(rx), torch.sin(rx)
        cy, sy = torch.cos(ry), torch.sin(ry)
        cz, sz = torch.cos(rz), torch.sin(rz)
        Rx = torch.tensor([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = torch.tensor([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rz = torch.tensor([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        return Rz @ Ry @ Rx

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() >= self.p:
            return x

        # Sample rotation angles (radians) and translation
        deg = self.rot_deg * 3.14159265 / 180.0
        rx, ry, rz = (torch.rand(3) * 2 - 1) * deg
        tx, ty, tz = (torch.rand(3) * 2 - 1) * self.trans_frac

        R = self._rot_matrix(rx, ry, rz)              # (3, 3)
        theta = torch.zeros(1, 3, 4)
        theta[0, :, :3] = R
        theta[0, :,  3] = torch.tensor([tx, ty, tz])

        x5 = x.unsqueeze(0)                           # (1, 1, D, H, W)
        grid = F.affine_grid(theta, x5.shape, align_corners=False)
        out  = F.grid_sample(
            x5, grid, mode="bilinear",
            padding_mode="zeros", align_corners=False,
        )
        return out.squeeze(0).clamp(0, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Composition
# ─────────────────────────────────────────────────────────────────────────────

class Compose:
    """Apply a list of transforms in order."""

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            x = t(x)
        return x

    def __repr__(self) -> str:
        names = ", ".join(type(t).__name__ for t in self.transforms)
        return f"Compose([{names}])"


# ─────────────────────────────────────────────────────────────────────────────
# Public factory
# ─────────────────────────────────────────────────────────────────────────────

def build_train_transform(strong: bool = False) -> Compose:
    """
    Build a training-time augmentation pipeline.

    Args:
        strong: If True, use wider parameter ranges for under-represented
                tracers (NAV n=85, PIB). If False, use standard ranges.

    Returns:
        Compose pipeline that maps (1, D, H, W) → (1, D, H, W) on [0, 1].
    """
    if strong:
        return Compose([
            RandFlipLR(p=0.5),
            RandAffine3D(rot_deg=10.0, trans_frac=0.08, p=0.7),
            RandGamma(r=0.20, p=0.7),
            RandBiasShift(s=0.08, p=0.5),
            RandGaussianNoise(sigma_max=0.04, p=0.5),
        ])

    return Compose([
        RandFlipLR(p=0.5),
        RandAffine3D(rot_deg=5.0, trans_frac=0.05, p=0.5),
        RandGamma(r=0.10, p=0.5),
        RandBiasShift(s=0.05, p=0.3),
        RandGaussianNoise(sigma_max=0.02, p=0.3),
    ])
