"""
mygo_centiloid — Amyloid β-PET Centiloid Prediction package.

Public API (all callers should import from here, not from submodules):

    from mygo_centiloid import PETResNet, BaselineCNN           # submission
    from mygo_centiloid import PETResNetNoGAP                   # −GAP ablation
    from mygo_centiloid import PETResNetNoFiLM                  # −FiLM ablation
    from mygo_centiloid import PETResNetTracerNormOnly          # TN-only ablation
    from mygo_centiloid import PETDataset                       # data
    from mygo_centiloid import build_train_transform            # augmentation
    from mygo_centiloid import CentiloidLoss, get_criterion     # losses
"""

from mygo_centiloid.data.dataset      import PETDataset
from mygo_centiloid.data.augmentation import build_train_transform
from mygo_centiloid.model.petresnet_film             import PETResNet, BaselineCNN
from mygo_centiloid.model.petresnet_no_gap           import PETResNetNoGAP
from mygo_centiloid.model.petresnet_no_film          import PETResNetNoFiLM
from mygo_centiloid.model.petresnet_tracer_norm_only import PETResNetTracerNormOnly
from mygo_centiloid.losses.losses                    import CentiloidLoss, get_criterion

__all__ = [
    "PETDataset",
    "build_train_transform",
    "PETResNet",
    "PETResNetNoGAP",
    "PETResNetNoFiLM",
    "PETResNetTracerNormOnly",
    "BaselineCNN",
    "CentiloidLoss",
    "get_criterion",
]

__version__ = "0.1.0"
