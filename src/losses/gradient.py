"""Smoothness regularizer penalizing the spatial gradient of a displacement field.

Author: Fl0rian
"""
from __future__ import absolute_import

import torch
import torch.nn as nn


class Grad3d(nn.Module):
    """L1 or L2 penalty on the finite-difference spatial gradient of a 3D displacement field."""

    def __init__(self, penalty: str = 'l1') -> None:
        """
        Args:
            penalty (str): Gradient penalty type, either 'l1' or 'l2'.
        """
        super().__init__()
        if penalty not in ['l1', 'l2']:
            raise ValueError(f"Unknown penalty type: {penalty}")
        self.penalty = penalty

    def forward(self, x_pred: torch.Tensor) -> torch.Tensor:
        """Compute the mean gradient penalty of ``x_pred``, shape (B, C, D, H, W)."""
        # Compute gradients in each direction
        dx = torch.abs(x_pred[:, :, 1:, :, :] - x_pred[:, :, :-1, :, :])
        dy = torch.abs(x_pred[:, :, :, 1:, :] - x_pred[:, :, :, :-1, :])
        dz = torch.abs(x_pred[:, :, :, :, 1:] - x_pred[:, :, :, :, :-1])
        # Apply penalty (squared for L2 penalty)
        if self.penalty == 'l2':
            dy, dx, dz = dy**2, dx**2, dz**2
        grad = (torch.mean(dx) + torch.mean(dy) + torch.mean(dz)) / 3.0
        return grad
