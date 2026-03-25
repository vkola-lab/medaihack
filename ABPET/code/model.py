"""
Model architecture for Amyloid PET Centiloid Prediction.

Baseline 3D CNN with tracer conditioning.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv3d -> BatchNorm -> ReLU -> MaxPool."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
        )

    def forward(self, x):
        return self.block(x)


class BaselineCNN(nn.Module):
    """
    Simple 3D CNN for centiloid regression.

    Architecture:
        4 conv blocks:  1 -> 32 -> 64 -> 128 -> 256  (each halves spatial dims)
        Global average pool -> 256-dim feature vector
        Concatenate with tracer embedding (8-dim)
        MLP head -> scalar prediction

    Input:  (B, 1, 128, 128, 128)
    Output: (B,)  predicted centiloid scores
    """

    def __init__(self, num_tracers: int, emb_dim: int = 8, mean_centiloid: float = 0.0):
        super().__init__()

        self.encoder = nn.Sequential(
            ConvBlock(1, 32),    # -> (B, 32, 64, 64, 64)
            ConvBlock(32, 64),   # -> (B, 64, 32, 32, 32)
            ConvBlock(64, 128),  # -> (B, 128, 16, 16, 16)
            ConvBlock(128, 256), # -> (B, 256, 8, 8, 8)
        )

        self.gap = nn.AdaptiveAvgPool3d(1)  # -> (B, 256, 1, 1, 1)

        self.tracer_emb = nn.Embedding(num_tracers, emb_dim)

        self.head = nn.Sequential(
            nn.Linear(256 + emb_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

        # Initialize final bias to dataset mean for faster convergence
        nn.init.constant_(self.head[-1].bias, mean_centiloid)

    def forward(self, x, tracer_idx):
        features = self.encoder(x)
        features = self.gap(features).flatten(1)             # (B, 256)
        tracer_features = self.tracer_emb(tracer_idx)        # (B, emb_dim)
        combined = torch.cat([features, tracer_features], 1) # (B, 264)
        return self.head(combined).squeeze(1)                # (B,)
