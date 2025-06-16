import torch
import torch.nn as nn
from typing import Union, Sequence


def compute_jacobian_determinant_3d(displacement, spacing=(1.0, 1.0, 1.0)):
    """
    Compute the Jacobian determinant of a 3D displacement field.

    Parameters:
    - displacement: torch.Tensor of shape (3, D, H, W), representing the displacement field.
    - spacing: tuple of floats (dz, dy, dx), representing the physical spacing between voxels.

    Returns:
    - jacobian_determinant: torch.Tensor of shape (D, H, W), the Jacobian determinant at each point.
    """
    dz, dy, dx = spacing
    du_dz = torch.gradient(displacement[2], spacing=dz, dim=0)[0]
    du_dy = torch.gradient(displacement[2], spacing=dy, dim=1)[0]
    du_dx = torch.gradient(displacement[2], spacing=dx, dim=2)[0]

    dv_dz = torch.gradient(displacement[1], spacing=dz, dim=0)[0]
    dv_dy = torch.gradient(displacement[1], spacing=dy, dim=1)[0]
    dv_dx = torch.gradient(displacement[1], spacing=dx, dim=2)[0]

    dw_dz = torch.gradient(displacement[0], spacing=dz, dim=0)[0]
    dw_dy = torch.gradient(displacement[0], spacing=dy, dim=1)[0]
    dw_dx = torch.gradient(displacement[0], spacing=dx, dim=2)[0]

    # Construct the Jacobian matrix components
    J11 = 1 + du_dx
    J12 = du_dy
    J13 = du_dz
    J21 = dv_dx
    J22 = 1 + dv_dy
    J23 = dv_dz
    J31 = dw_dx
    J32 = dw_dy
    J33 = 1 + dw_dz

    # Compute the determinant of the Jacobian matrix
    jacobian_determinant = (
        J11 * (J22 * J33 - J23 * J32) -
        J12 * (J21 * J33 - J23 * J31) +
        J13 * (J21 * J32 - J22 * J31)
    )

    return jacobian_determinant

def compute_jacobian_determinant(J):
    # Assume J is in normalized coordinates [-1, 1]
    # Step 1: Normalize from [-1, 1] → [0, 1]
    J = J + 1          #恢复到0到1
    J = J / 2.
    scale_factor = torch.tensor([J.size(1), J.size(2), J.size(3)]).to(J).view(1, 1, 1, 1, 3) * 1.
    J = J * scale_factor

    dy = J[:, 1:, :-1, :-1, :] - J[:, :-1, :-1, :-1, :]
    dx = J[:, :-1, 1:, :-1, :] - J[:, :-1, :-1, :-1, :]
    dz = J[:, :-1, :-1, 1:, :] - J[:, :-1, :-1, :-1, :]

    Jdet0 = dx[:, :, :, :, 0] * (dy[:, :, :, :, 1] * dz[:, :, :, :, 2] - dy[:, :, :, :, 2] * dz[:, :, :, :, 1])
    Jdet1 = dx[:, :, :, :, 1] * (dy[:, :, :, :, 0] * dz[:, :, :, :, 2] - dy[:, :, :, :, 2] * dz[:, :, :, :, 0])
    Jdet2 = dx[:, :, :, :, 2] * (dy[:, :, :, :, 0] * dz[:, :, :, :, 1] - dy[:, :, :, :, 1] * dz[:, :, :, :, 0])

    Jdet = Jdet0 - Jdet1 + Jdet2
    return Jdet


def jacobian_determinant_3d(deformed_grid: torch.Tensor) -> torch.Tensor:
    """
    Computes the determinant of the Jacobian numerically, given the deformed
    output grid and returns the percentage of negative values

    Args:
        deformed_grid (torch.Tensor): [B, D, H, W, 3]

    Returns:
        torch.Tensor: the percentage of negative determinants
    """
    dy = deformed_grid[:, 1:, :-1, :-1, :] - deformed_grid[:, :-1, :-1, :-1, :]
    dx = deformed_grid[:, :-1, 1:, :-1, :] - deformed_grid[:, :-1, :-1, :-1, :]
    dz = deformed_grid[:, :-1, :-1, 1:, :] - deformed_grid[:, :-1, :-1, :-1, :]

    det0 = dx[:, :, :, :, 0] * (dy[:, :, :, :, 1] * dz[:, :, :, :, 2] - dy[:, :, :, :, 2] * dz[:, :, :, :, 1])
    det1 = dx[:, :, :, :, 1] * (dy[:, :, :, :, 0] * dz[:, :, :, :, 2] - dy[:, :, :, :, 2] * dz[:, :, :, :, 0])
    det2 = dx[:, :, :, :, 2] * (dy[:, :, :, :, 0] * dz[:, :, :, :, 1] - dy[:, :, :, :, 1] * dz[:, :, :, :, 0])

    determinants = det0 - det1 + det2

    return determinants


class Jacobianloss(nn.Module):
    """
    Jacobian loss for penalizing the Jacobian determinant of a displacement field.
    This loss can be used to ensure that the transformation is invertible and smooth.
    """
    def __init__(self):
        super(Jacobianloss, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Penalizing Jacobian
        Args:
            x (torch.Tensor): Displacement field of shape (B, C, D, H, W) or (B, C, H, W).
            spacing (Union[list[int], tuple[int]]): Spacing of the input image.
        Returns:
            torch.Tensor: Jacobian loss value.
        '''

        Jdet = compute_jacobian_determinant(x)
        Neg_Jac = 0.5 * (torch.abs(Jdet) - Jdet)
        return torch.sum(Neg_Jac)
