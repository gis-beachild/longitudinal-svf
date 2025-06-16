import monai
import torch
import torch.nn as nn
from torch import Tensor
from src.losses.similarity.normalized_cross_correlation_loss import NormalizedCrossCorrelationLoss
from src.losses.regularisation.jacobian import Jacobianloss
import torch.nn.functional as F

class PairwiseRegistrationLoss(nn.Module):
    def __init__(self, seg_loss: nn.Module = None, mag_loss: nn.Module= None,
                 grad_loss: nn.Module= None, lambda_sim: float = 0, lambda_seg: float = 0,
                 lambda_mag: float = 0, lambda_grad: float = 0):
        super().__init__()
        self.sim_loss = nn.MSELoss()
        self.grad_loss = grad_loss
        self.mag_loss = mag_loss
        self.seg_loss = seg_loss
        self.lambda_seg = lambda_seg
        self.lambda_sim = lambda_sim
        self.lambda_mag = lambda_mag
        self.lambda_grad = lambda_grad

    def forward(self, source_image: torch.Tensor, target_image: torch.Tensor, source_label: torch.Tensor,
                target_label: torch.Tensor, f: torch.Tensor, bf: torch.Tensor, v: torch.Tensor = None) -> torch.Tensor:
        '''
        Compute the registration loss for a pair of subjects.
        :param source_image: Source image
        :param target_image: Target image
        :param source_label: Source label
        :param target_label: Target label
        :param f: Flow field
        :param bf: Backward flow field
        :param v: Velocity field (Set to None to penalize the displacement field)
        '''
        loss_errors = torch.zeros(4).float().to(source_image.device)
        # Similarity loss
        if self.lambda_sim > 0:
            loss_errors[0] = self.lambda_sim * self.sim_loss(source_image, target_image)

        # Segmentation loss
        if self.lambda_seg > 0:
            loss_errors[1] = self.lambda_seg * F.mse_loss(source_label, target_label, reduction='none').mean(dim=(0, 2, 3, 4)).sum()

        # Magnitude loss
        if self.lambda_mag > 0:
            loss_errors[2] = self.lambda_mag * (self.mag_loss(f) if v is None else self.mag_loss(v))

        # Gradient loss
        if self.lambda_grad > 0:
            loss_errors[3] = self.lambda_grad * (self.grad_loss(f) if v is None else self.grad_loss(v))

        return loss_errors