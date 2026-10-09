"""Sinusoidal positional encodings and an implicit neural representation (INR) of a 3D+time flow field.

Author: Fl0rian
"""
from typing import List

import torch
import torch.nn as nn

from .mlp import MLP


def time_encoding(t: torch.Tensor, max_freq: int = 10) -> torch.Tensor:
    '''
    Positional encoding for time.
    :param t: Time to encode
    :param max_freq: Maximum frequency for encoding
    '''
    encoded_time = []
    freqs = 2 ** torch.arange(max_freq, dtype=torch.float32)
    for freq in freqs:
        encoded_time.append(torch.sin(t * freq))
        encoded_time.append(torch.cos(t * freq))
    return torch.cat(encoded_time, dim=-1).to(t.device)


def positional_encoding_3d(coords: torch.Tensor, max_freq: int = 10) -> torch.Tensor:
    '''
    Positional encoding for 3D coordinates.
    :param coords: Coordinates to encode
    :param max_freq: Maximum frequency for encoding
    '''
    encoded_positions = []
    freqs = 2 ** torch.arange(max_freq, dtype=torch.float32)
    for i in range(coords.shape[1]):
        pos = coords[:, i:i + 1]
        encoded = []
        for freq in freqs:
            encoded.append(torch.sin(pos * freq))
            encoded.append(torch.cos(pos * freq))
        encoded_positions.append(torch.cat(encoded, dim=-1))
    return torch.cat(encoded_positions, dim=-1).to(coords.device)


class ImplicitNeuralNetwork(nn.Module):
    '''
        Maps a scalar time value to a dense 3D flow field over a fixed volume shape,
        by evaluating an MLP over positionally-encoded (x, y, z, t) coordinates.
    '''
    def __init__(self, inshape: List[int], out_channels: int, hidden_dim: int = 32, max_freq: int = 10) -> None:
        '''
        :param inshape: Spatial shape (D, H, W) of the volume to generate a flow field for.
        :param out_channels: Number of output channels (flow components) per voxel.
        :param hidden_dim: Hidden width of the underlying MLP.
        :param max_freq: Number of frequency bands used in the positional encoding.
        '''
        super(ImplicitNeuralNetwork, self).__init__()
        self.max_freq = max_freq
        self.inshape = inshape
        self.model = MLP(input_dim=(len(inshape) + 1) * self.max_freq * 2, output_dim=out_channels, hidden_dim=hidden_dim)
        ranges = [torch.linspace(0, s - 1, s) for s in inshape]
        grids = torch.meshgrid(*ranges)
        coords = torch.stack([grid.flatten() for grid in grids], dim=-1)
        encoded_coords = positional_encoding_3d(coords=coords, max_freq=self.max_freq)
        self.register_buffer('encoded_coords', encoded_coords)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the implicit flow field at scalar time ``x``, returning a (1, 3, D, H, W) tensor."""
        x = positional_encoding_3d(torch.full((self.encoded_coords.shape[0], 1), x.item()), max_freq=self.max_freq).to(self.encoded_coords.device)
        encoded_inputs = torch.cat([self.encoded_coords, x], dim=-1)  # Combine (x, y, z, t)
        y = self.model(encoded_inputs)
        y = y.view(self.inshape[0], self.inshape[1], self.inshape[2], 3).permute(3, 0, 1, 2).unsqueeze(0)
        return y
