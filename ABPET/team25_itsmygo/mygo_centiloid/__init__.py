"""
mygo_centiloid — Amyloid β-PET Centiloid Prediction package.

Public API (all callers should import from here, not from submodules):

    from mygo_centiloid import mygo_centiloid, BaselineCNN          # architecture
    from mygo_centiloid import PETDataset                       # data
    from mygo_centiloid import build_train_transform            # augmentation
    from mygo_centiloid import CentiloidLoss, get_criterion     # losses
"""

from mygo_centiloid.data.dataset      import PETDataset
from mygo_centiloid.data.augmentation import build_train_transform
from mygo_centiloid.model.petresnet_film          import PETResNet, BaselineCNN
from mygo_centiloid.model.petresnet_no_film  import PETResNetNoFiLM
from mygo_centiloid.model.petresnet_attn     import PETResNetAttn, SpatialAttention3D
from mygo_centiloid.model.petresnet_attn_gated import PETResNetAttnGated, SpatialAttention3DGated
from mygo_centiloid.losses.losses             import CentiloidLoss, get_criterion

__all__ = [
    "PETDataset",
    "build_train_transform",
    "PETResNet",
    "PETResNetNoFiLM",
    "PETResNetAttn",
    "SpatialAttention3D",
    "PETResNetAttnGated",
    "SpatialAttention3DGated",
    "BaselineCNN",
    "CentiloidLoss",
    "get_criterion",
]

__version__ = "0.1.0"
