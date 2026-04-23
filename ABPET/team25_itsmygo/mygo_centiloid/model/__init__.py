"""Model architecture subpackage.

Exposes the competition submission (`PETResNet`) and the three factorial
ablation variants reported in §3.5 / §4.3 of the MYGO-Centiloid report:

    PETResNet (submission)     TracerNorm + FiLM + GAP concat       11.79 CL
    PETResNetNoGAP             TracerNorm + FiLM                    10.83 CL
    PETResNetNoFiLM            TracerNorm + GAP concat              10.49 CL
    PETResNetTracerNormOnly    TracerNorm                            9.03 CL
"""

from mygo_centiloid.model.petresnet_film import (
    PETResNet, BaselineCNN, TracerNorm, FiLMBlock, ResBlock3D,
)
from mygo_centiloid.model.petresnet_no_gap           import PETResNetNoGAP
from mygo_centiloid.model.petresnet_no_film          import PETResNetNoFiLM
from mygo_centiloid.model.petresnet_tracer_norm_only import PETResNetTracerNormOnly

__all__ = [
    "PETResNet", "BaselineCNN", "TracerNorm", "FiLMBlock", "ResBlock3D",
    "PETResNetNoGAP",
    "PETResNetNoFiLM",
    "PETResNetTracerNormOnly",
]
