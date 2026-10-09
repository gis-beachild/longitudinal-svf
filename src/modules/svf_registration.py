"""Wraps a deformation-predicting backbone with stationary-velocity-field integration for registration.

Author: Fl0rian
"""

import monai.networks.blocks
import torch
import torch.nn as nn
from torch import Tensor


class SVFRegistrationModule(nn.Module):
    """
        Registration module for 3D image registration.yaml with DVF
    """
    def __init__(self, model: nn.Module, int_steps: int = 7) -> None:
        """
        :param model: nn.Module predicting a stationary velocity field (DVF) from the input images.
        :param int_steps: Number of scaling-and-squaring integration steps used to turn the DVF into a DDF.
        """
        super().__init__()
        self.model = model
        # Vector integration based on Runge-Kutta method
        self.dvf2ddf = monai.networks.blocks.DVF2DDF(num_steps=int_steps, mode='bilinear', padding_mode='zeros') # type: ignore 

    def forward(self, data: Tensor) -> Tensor:
        '''
            Forward pass of the registration module
            :param data: Input images
            :return: Deformation field
        '''
        return self.model(data)


    def load_network(self, path: str) -> None:
        '''
            Load the network weights
            :param path: Path to the weights
        '''
        self.model.load_state_dict(torch.load(path))

    def velocity2displacement(self, dvf: Tensor) -> Tensor:
        '''
            Convert the velocity field to a flow field
            :param dvf: Velocity field
            :return: Deformation field
        '''
        return self.dvf2ddf(dvf)
