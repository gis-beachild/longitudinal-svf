import monai.networks.blocks
import torch
import torch.nn as nn
from torch import Tensor
from monai import networks
import torch.nn.functional as F
from typing import Union
import numpy as np

def warp(image: Tensor, flow: Tensor, mode: str = 'bilinear') -> Tensor:
    """
    Warp an image using a dense flow field.

    Args:
        image (Tensor): Input image tensor of shape (B, C, H, W) or (B, C, D, H, W).
        flow (Tensor): Flow field of the same shape as image (excluding channel), shape (B, D, H, W, 3) or (B, H, W, 2).
        mode (str): Interpolation mode ('bilinear' or 'nearest').

    Returns:
        Tensor: Warped image.
    """
    warp = monai.networks.blocks.Warp(mode=mode, padding_mode='reflection')
    return warp(image, flow)



def compose(flow_a: torch.Tensor, flow_b: torch.Tensor) -> torch.Tensor:
    warp = monai.networks.blocks.Warp(mode='bilinear', padding_mode='reflection')
    return warp(flow_a, flow_b) + flow_b



def displacement2grid(flow: Tensor, grid_normalize=False) -> torch.Tensor:
    """
    Convert a flow field to a normalized sampling grid (phi) for grid_sample.

    Args:
        :param flow : Tensor - Flow field of shape (B, 3, D, H, W)

    Returns:
        Tensor: Normalized grid (phi) suitable for F.grid_sample.
        :param grid_normalize:
    """
    grid = _make_identity_grid(flow.shape).to(flow.device)
    phi = flow + grid
    shape = flow.shape[2:]
    if grid_normalize:
        for i in range(len(shape)):
            phi[:, i, ...] = phi[:, i, ...] * 2 / (shape[i] - 1) - 1

    # move channels dim to last position
    return phi

def _make_identity_grid(shape: torch.Tensor.shape) -> torch.Tensor:
    spatial_dims = shape[2:]
    coords = [torch.arange(0, s) for s in spatial_dims]
    mesh = torch.meshgrid(*coords, indexing='ij')  # Ensures correct axis order
    grid = torch.stack(mesh, dim=0).float()  # (C, ...)
    grid = grid.unsqueeze(0)  # Add batch dimension: (1, C, ...)
    return grid

