"""Penalty on the raw magnitude of a tensor (e.g. a velocity/displacement field), pulling it toward zero.

Author: Fl0rian
"""
from __future__ import absolute_import

import torch
import torch.nn as nn


class MagnitudeLoss(nn.Module):
    """L1 or L2 penalty of ``x`` against the zero tensor."""

    def __init__(self, penalty: str = 'l1') -> None:
        """
        Args:
            penalty (str): Penalty type, either 'l1' or 'l2'.
        """
        super().__init__()
        loss_fn = {'l1': nn.L1Loss, 'l2': nn.MSELoss}
        if penalty not in loss_fn:
            raise ValueError(f"Unknown penalty type: {penalty}")
        self.loss = loss_fn[penalty]()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the penalty of ``x`` against zero."""
        return self.loss(x, torch.zeros_like(x))
