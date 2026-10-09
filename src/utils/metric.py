"""Topology-aware (clDice) and Jacobian-based metrics for evaluating deformation quality.

Author: Fl0rian
"""
import meshio
import torch
import torch.nn as nn
import numpy as np
from src.losses.jacobian import compute_jacobian_determinant
from skimage.morphology import skeletonize
from src.utils.gyrification_index import compute_gyrification_index, rescale_initial_smooth_mesh_to_folded_mesh

def cl_score(v, s):
    """[this function computes the skeleton volume overlap]

    Args:
        v ([bool]): [image]
        s ([bool]): [skeleton]

    Returns:
        [float]: [computed skeleton volume intersection]
    """
    return np.sum(v*s)/np.sum(s)


def clDice(v_p, v_l):
    """[this function computes the cldice metric]

    Args:
        v_p ([bool]): [predicted image]
        v_l ([bool]): [ground truth image]

    Returns:
        [float]: [cldice metric]
    """
    tprec = cl_score(v_p,skeletonize(v_l))
    tsens = cl_score(v_l,skeletonize(v_p))
    return 2*tprec*tsens/(tprec+tsens)

class NegativeJacobian(nn.Module):
    """Counts the number of voxels with a negative (folding) Jacobian determinant."""

    def __init__(self) -> None:
        super(NegativeJacobian, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Count negative-Jacobian voxels in displacement/grid field ``x``."""
        return (compute_jacobian_determinant(x) < 0).sum()

class LogJacobian(nn.Module):
    """Log of the (clamped, non-negative) Jacobian determinant."""

    def __init__(self) -> None:
        super(LogJacobian, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the log-Jacobian-determinant map of ``x``, clamped away from zero."""
        det = compute_jacobian_determinant(x)
        eps = 1e-8
        safe_det = torch.clamp(det, min=eps)
        log_det = torch.log(safe_det)
        return log_det


def GyrificationIndex(smooth_mesh_path: str, folded_mesh_path: str) -> float:
    """Computes the gyrification index of a surface mesh, defined as the ratio of the total surface area to the convex hull area."""
    smooth_mesh = meshio.read(smooth_mesh_path)
    folded_mesh = meshio.read(folded_mesh_path)
    # rescale initial smooth brain mesh onto the folded brain mesh
    rescaled_initial_smooth_mesh = rescale_initial_smooth_mesh_to_folded_mesh(smooth_mesh, folded_mesh)
    GI = compute_gyrification_index(rescaled_initial_smooth_mesh, folded_mesh)
    return GI
