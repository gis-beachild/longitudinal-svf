"""Jacobian-determinant computation and a regularizer penalizing negative (folding) determinants.

Author: Fl0rian
"""
from typing import Tuple

import torch
import torch.nn as nn


def compute_jacobian_determinant_3d(displacement: torch.Tensor, spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> torch.Tensor:
    """
    Compute the Jacobian determinant of a 3D displacement field.

    Parameters:
    - displacement: tensor of shape (3, D, H, W) or (B, 3, D, H, W).
      Components follow the tensor axes (D, H, W). With the default spacing,
      displacement is measured in voxels, as returned by MONAI's DVF2DDF.
    - spacing: spacing along (D, H, W). Use non-unit spacing only when the
      displacement components are measured in the same physical units.

    Returns:
    - jacobian_determinant: torch.Tensor of shape (D, H, W) or (B, D, H, W).
    """
    batched = displacement.ndim == 5
    if displacement.ndim == 4:
        displacement = displacement.unsqueeze(0)
    if displacement.ndim != 5 or displacement.shape[1] != 3:
        raise ValueError("displacement must have shape (3, D, H, W) or (B, 3, D, H, W)")

    d0_d, d0_h, d0_w = torch.gradient(displacement[:, 0], spacing=spacing, dim=(1, 2, 3))
    d1_d, d1_h, d1_w = torch.gradient(displacement[:, 1], spacing=spacing, dim=(1, 2, 3))
    d2_d, d2_h, d2_w = torch.gradient(displacement[:, 2], spacing=spacing, dim=(1, 2, 3))

    # Construct the Jacobian matrix components
    J11 = 1 + d0_d
    J12 = d0_h
    J13 = d0_w
    J21 = d1_d
    J22 = 1 + d1_h
    J23 = d1_w
    J31 = d2_d
    J32 = d2_h
    J33 = 1 + d2_w

    # Compute the determinant of the Jacobian matrix
    jacobian_determinant = (
        J11 * (J22 * J33 - J23 * J32) -
        J12 * (J21 * J33 - J23 * J31) +
        J13 * (J21 * J32 - J22 * J31)
    )

    return jacobian_determinant if batched else jacobian_determinant[0]

def compute_jacobian_determinant(J: torch.Tensor) -> torch.Tensor:
    """Determinant of a normalized grid ``(B, D, H, W, 3)``.

    Components are ordered ``(D, H, W)`` as in ``displacement2grid``.
    Forward differences give an output of shape ``(B, D-1, H-1, W-1)``.
    """
    if J.ndim != 5 or J.shape[-1] != 3 or any(size < 2 for size in J.shape[1:4]):
        raise ValueError("grid must have shape (B, D, H, W, 3), with D, H, W >= 2")

    d_d = J[:, 1:, :-1, :-1] - J[:, :-1, :-1, :-1]
    d_h = J[:, :-1, 1:, :-1] - J[:, :-1, :-1, :-1]
    d_w = J[:, :-1, :-1, 1:] - J[:, :-1, :-1, :-1]
    determinant = (
        d_d[..., 0] * (d_h[..., 1] * d_w[..., 2] - d_h[..., 2] * d_w[..., 1])
        - d_h[..., 0] * (d_d[..., 1] * d_w[..., 2] - d_d[..., 2] * d_w[..., 1])
        + d_w[..., 0] * (d_d[..., 1] * d_h[..., 2] - d_d[..., 2] * d_h[..., 1])
    )
    # Each normalized component spans [-1, 1] over size-1 voxel intervals.
    return determinant * ((J.shape[1] - 1) * (J.shape[2] - 1) * (J.shape[3] - 1) / 8)


def jacobian_determinant_3d(deformed_grid: torch.Tensor) -> torch.Tensor:
    """Return the determinant map of a normalized deformed sampling grid."""
    return compute_jacobian_determinant(deformed_grid)


class Jacobianloss(nn.Module):
    """
    Jacobian loss for penalizing the Jacobian determinant of a displacement field.
    This loss can be used to ensure that the transformation is invertible and smooth.
    """
    def __init__(self) -> None:
        super(Jacobianloss, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Penalizing Jacobian
        Args:
            x (torch.Tensor): Displacement field of shape (B, 3, D, H, W).
        Returns:
            torch.Tensor: Jacobian loss value.
        '''

        Jdet = compute_jacobian_determinant_3d(x)
        Neg_Jac = 0.5 * (torch.abs(Jdet) - Jdet)
        return torch.sum(Neg_Jac)
