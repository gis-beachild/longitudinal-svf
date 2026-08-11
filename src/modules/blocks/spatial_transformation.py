"""Spatial transformer and scaling-and-squaring flow integration, adapted from the VoxelMorph repo.

Author: Fl0rian
"""

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

# code from voxelmorph repo


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """
    def __init__(self, size: Sequence[int], mode: str = 'bilinear') -> None:
        """
        Args:
            size (Sequence[int]): Spatial shape (D, H, W) or (H, W) of the volumes to be resampled.
            mode (str): Interpolation mode passed to ``F.grid_sample``.
        """
        super().__init__()
        self.mode = mode
        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src (torch.Tensor): Source volume/image to resample, shape (B, C, *spatial_shape).
            flow (torch.Tensor): Displacement field, shape (B, spatial_dims, *spatial_shape).

        Returns:
            torch.Tensor: ``src`` resampled at ``grid + flow``, shape (B, C, *spatial_shape).
        """
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]
        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
        # move channels dim to last position
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]
        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


#from voxelmorph repo
class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """
    def __init__(self, inshape: Sequence[int], nsteps: int) -> None:
        """
        Args:
            inshape (Sequence[int]): Spatial shape (D, H, W) or (H, W) of the velocity field.
            nsteps (int): Number of scaling-and-squaring steps; the field is first scaled by 2**-nsteps.
        """
        super().__init__()
        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec: torch.Tensor) -> torch.Tensor:
        """Integrate a stationary velocity field into a displacement field via scaling and squaring.

        Args:
            vec (torch.Tensor): Stationary velocity field, shape (B, spatial_dims, *spatial_shape).

        Returns:
            torch.Tensor: Integrated displacement field, same shape as ``vec``.
        """
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec
