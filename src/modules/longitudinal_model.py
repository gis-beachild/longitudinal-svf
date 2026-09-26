"""SVF-based longitudinal deformation model: a registration backbone plus an optional monotonic time warp.

Author: Fl0rian
"""
import os

from src.modules.svf_registration import SVFRegistrationModule
import torch
import torch.nn as nn

from modules.monotonic_mlp import MonotonicMLP


class LongitudinalDeformation(nn.Module):
    """Wraps a stationary-velocity-field registration model with a per-subject time encoding (linear or monotonic MLP)."""

    def __init__(self, svf_model : SVFRegistrationModule, time_mode: str, t0: int, t1: int) -> None:
        '''
        Our longitudinal deformation model
        :param svf_model: Registration model
        :param time_mode: Interpolation mode, either 'mlp' or 'linear'
        :param t0: time 0
        :param t1: time 1
        '''
        super().__init__()
        self.t0 = t0
        self.t1 = t1
        self.svf_model = svf_model
        self.time_mode = time_mode
        self.mlp_model = None
        if self.time_mode == 'mlp':
            self.mlp_model = MonotonicMLP()

    def forward(self, data : torch.Tensor) -> torch.Tensor:
        """Predict the stationary velocity field for the input images via the wrapped registration model."""
        return self.svf_model(data)

    def encode_time(self, time: torch.Tensor) -> torch.Tensor:
        """Map a normalized time value in [0, 1] through the temporal model (identity when ``time_mode == 'linear'``)."""
        if self.time_mode == 'mlp' and self.mlp_model is not None:
            time =  self.mlp_model(time)
        return time

    def load_reg_model(self, path: str) -> None:
        """Load the registration backbone's weights from ``path``."""
        self.svf_model.load_state_dict(torch.load(path))

    def load_temporal(self, path: str) -> None:
        """Load the temporal MLP's weights from ``path`` (no-op if ``time_mode`` is not 'mlp')."""
        if self.mlp_model is not None:
            self.mlp_model.load_state_dict(torch.load(path))

    def save(self, path: str) -> None:
        """Save the registration backbone's (and, if present, temporal MLP's) weights under directory ``path``."""
        torch.save(self.svf_model.state_dict(), os.path.join(path, 'model.pth'))
        if self.mlp_model is not None:
            torch.save(self.mlp_model.state_dict(), os.path.join(path, 'temporal_model.pth'))
