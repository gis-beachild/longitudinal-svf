"""Helpers for warping images/flows and converting between displacement fields and sampling grids.

Author: Fl0rian
"""
import io
from typing import Tuple

import torch
from torch import Tensor
import monai.networks.blocks
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from PIL import Image
from monai.networks.utils import meshgrid_ij


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
    warp = monai.networks.blocks.Warp(mode=mode, padding_mode='reflection') # type: ignore
    return warp(image, flow) 


def compose(flow_a: Tensor, flow_b: Tensor) -> Tensor:
    """
    Compose two displacement fields via ``flow_a`` warped by ``flow_b``, added to ``flow_b``.

    Args:
        flow_a (Tensor): Displacement field to be warped, shape (B, C, D, H, W) or (B, C, H, W).
        flow_b (Tensor): Displacement field used to warp ``flow_a``, same shape as ``flow_a``.

    Returns:
        Tensor: Composed displacement field, same shape as the inputs.
    """
    warp = monai.networks.blocks.Warp(mode='bilinear', padding_mode='reflection') # type: ignore
    return warp(flow_a, flow_b) + flow_b


def get_reference_grid(ddf: Tensor) -> Tensor:
    """
    Build an integer-valued reference sampling grid matching the spatial shape of a displacement field.

    Args:
        ddf (Tensor): Displacement field of shape (B, spatial_dims, *spatial_shape).

    Returns:
        Tensor: Reference grid of shape (B, spatial_dims, *spatial_shape), broadcast over the batch dimension.
    """
    mesh_points = [torch.arange(0, dim) for dim in ddf.shape[2:]]
    grid = torch.stack(meshgrid_ij(*mesh_points), dim=0)  # (spatial_dims, ...)
    grid = torch.stack([grid] * ddf.shape[0], dim=0)  # (batch, spatial_dims, ...)
    ref_grid = grid.to(ddf)
    return ref_grid


def displacement2grid(flow: Tensor) -> Tensor:
    """
    Convert a flow field to a normalized sampling grid (phi) for grid_sample.

    Args:
        flow (Tensor): Flow field of shape (B, 3, D, H, W) or (B, 2, H, W).

    Returns:
        Tensor: Normalized grid (phi) suitable for F.grid_sample, shape (B, *spatial_shape, spatial_dims).
    """
    spatial_dims = len(flow.shape) - 2
    if spatial_dims not in (2, 3):
        raise NotImplementedError(f"got unsupported spatial_dims={spatial_dims}, currently support 2 or 3.")
    grid = get_reference_grid(flow).to(flow.device) + flow

    grid = grid.permute([0] + list(range(2, 2 + spatial_dims)) + [1])
    for i, dim in enumerate(grid.shape[1:-1]):
        grid[..., i] = grid[..., i] * 2 / (dim - 1) - 1
    return grid


def _make_identity_grid(shape: torch.Size) -> Tensor:
    """
    Build an unnormalized identity sampling grid (integer voxel coordinates) for a given tensor shape.

    Args:
        shape (torch.Size): Shape of the reference tensor, formatted as (B, C, *spatial_shape).

    Returns:
        Tensor: Identity grid of shape (1, spatial_dims, *spatial_shape).
    """
    spatial_dims = shape[2:]
    coords = [torch.arange(0, s) for s in spatial_dims]
    mesh = torch.meshgrid(*coords, indexing='ij')  # Ensures correct axis order
    grid = torch.stack(mesh, dim=0).float()  # (C, ...)
    grid = grid.unsqueeze(0)  # Add batch dimension: (1, C, ...)
    return grid


def plt_grid(xy: Tensor, ratio: float = 0.8) -> Tuple[Image.Image, Figure]:
    """
    Plots the 2D grid
    Args:
        xy (torch.Tensor): generated grids [h, w]
        ratio (float): width-to-height ratio of the rendered figure.

    Returns:
        Tuple[Image.Image, Figure]: the rendered grid as a PIL image, and the underlying matplotlib figure.
    """

    # Figure size in inches = pixels / DPI
    dpi = 100
    width_px = int(1000 * ratio)
    height_px = 1000

    # Create figure with exact size
    dpi = 100
    fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    ax = fig.add_axes((0, 0, 1, 1))  # Fill entire canvas

    # Set black background
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Draw grid lines in white
    for i in range(xy.shape[0]):
        ax.plot(xy[i, :, 0], xy[i, :, 1], '-', lw=1.3, color='k')
    for j in range(xy.shape[1]):
        ax.plot(xy[:, j, 0], xy[:, j, 1], '-', lw=1.3, color='k')

    ax.set_xlim(xy[..., 0].min().item(), xy[..., 0].max().item())
    ax.set_ylim(xy[..., 1].min().item(), xy[..., 1].max().item())
    ax.invert_yaxis()  # Optional, depends on how your grid is laid out

    # Hide all axis decorations
    ax.axis('off')

    # Save to a PIL Image using in-memory buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches=None, pad_inches=0, transparent=False)
    plt.close(fig)
    buf.seek(0)
    image = Image.open(buf).convert('RGB')
    buf.close()

    return image, fig
