"""3D smoothing kernels: a fixed separable Gaussian convolution and a box-window average via summed-area tables.

Author: Fl0rian
"""

from typing import Tuple

import torch
import numpy as np
import torch.nn.functional as F
import scipy.stats as st


class GaussianKernel(torch.nn.Module):
    """Separable 3D Gaussian smoothing filter with a fixed, non-trainable kernel."""

    def __init__(self, win: int = 11, nsig: float = 0.1) -> None:
        """
        Args:
            win (int): Kernel window size (number of taps) along each spatial axis.
            nsig (float): Half-width, in standard deviations, of the Gaussian sampled over ``win`` taps.
        """
        super(GaussianKernel, self).__init__()
        self.win = win
        self.nsig = nsig
        kernel_x, kernel_y, kernel_z = self.gkern1D_xyz(self.win, self.nsig)
        kernel = kernel_x * kernel_y * kernel_z
        self.register_buffer("kernel_x", kernel_x)
        self.register_buffer("kernel_y", kernel_y)
        self.register_buffer("kernel_z", kernel_z)
        self.register_buffer("kernel", kernel)

    def gkern1D(self, kernlen: int = None, nsig: float = None) -> torch.Tensor:
        """
        :param kernlen: number of taps in the returned 1D kernel.
        :param nsig: large nsig gives more freedom(pixels as agents), small nsig is more fluid.
        :return: Returns a 1D Gaussian kernel.
        """
        x = np.linspace(-nsig, nsig, kernlen + 1)
        kern1d = np.diff(st.norm.cdf(x))
        kern1d = kern1d / kern1d.sum()
        return torch.tensor(kern1d, requires_grad=False).float()

    def gkern1D_xyz(self, kernlen: int = None, nsig: float = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns 3 1D Gaussian kernel on xyz direction."""
        kernel_1d = self.gkern1D(kernlen, nsig)
        kernel_x = kernel_1d.view(1, 1, -1, 1, 1)
        kernel_y = kernel_1d.view(1, 1, 1, -1, 1)
        kernel_z = kernel_1d.view(1, 1, 1, 1, -1)
        return kernel_x, kernel_y, kernel_z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the separable 3D Gaussian filter to a (B, C, D, H, W) tensor."""
        pad = int((self.win - 1) / 2)
        # Apply Gaussian by 3D kernel
        x = F.conv3d(x, self.kernel, padding=pad)
        return x


class AveragingKernel(torch.nn.Module):
    """3D box-window average computed efficiently via a cumulative-sum (summed-area table) trick."""

    def __init__(self, win: int = 11) -> None:
        """
        Args:
            win (int): Side length of the cubic averaging window.
        """
        super(AveragingKernel, self).__init__()
        self.win = win

    def window_averaging(self, v: torch.Tensor) -> torch.Tensor:
        """Compute the local mean of ``v`` over a ``win``-sized cubic window at every voxel.

        Args:
            v (torch.Tensor): Input tensor of shape (B, C, D, H, W).

        Returns:
            torch.Tensor: Windowed average, same shape as ``v``.
        """
        win_size = self.win
        v = v.double()

        half_win = int(win_size / 2)
        pad = [half_win + 1, half_win] * 3

        v_padded = F.pad(v, pad=pad, mode='constant', value=0)  # [x+pad, y+pad, z+pad]

        # Run the cumulative sum across all 3 dimensions
        v_cs_x = torch.cumsum(v_padded, dim=2)
        v_cs_xy = torch.cumsum(v_cs_x, dim=3)
        v_cs_xyz = torch.cumsum(v_cs_xy, dim=4)

        x, y, z = v.shape[2:]

        # Use subtraction to calculate the window sum
        v_win = v_cs_xyz[:, :, win_size:, win_size:, win_size:] \
                - v_cs_xyz[:, :, win_size:, win_size:, :z] \
                - v_cs_xyz[:, :, win_size:, :y, win_size:] \
                - v_cs_xyz[:, :, :x, win_size:, win_size:] \
                + v_cs_xyz[:, :, win_size:, :y, :z] \
                + v_cs_xyz[:, :, :x, win_size:, :z] \
                + v_cs_xyz[:, :, :x, :y, win_size:] \
                - v_cs_xyz[:, :, :x, :y, :z]

        # Normalize by number of elements
        v_win = v_win / (win_size ** 3)
        v_win = v_win.float()
        return v_win

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """Apply the windowed average to a (B, C, D, H, W) tensor."""
        return self.window_averaging(v)
