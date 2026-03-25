"""
Loss functions for Amyloid PET Centiloid Prediction.

Add custom loss classes here and register them in get_criterion().
"""

import torch.nn as nn


def get_criterion(name: str = "mse", **kwargs):
    """Factory for loss functions.

    Args:
        name: One of "mse", "mae".
        **kwargs: Passed to the loss constructor.
    """
    if name == "mse":
        return nn.MSELoss()
    elif name == "mae":
        return nn.L1Loss()
    else:
        raise ValueError(f"Unknown loss: {name!r}. Choose from: mse, mae")
