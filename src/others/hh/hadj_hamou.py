import os
import itertools
import torch
import torch.nn as nn
import torchio as tio
import torch
import torch.nn as nn
import torch.nn.functional as F
from registration_svf.registration import RegistrationModule

class HadjHamouSVF(nn.Module):
    '''
    Implementation of Hadj-Hamou longitudinal deformation model
    '''
    def __init__(self, model : RegistrationModule, t0: int, t1: int, device):
        '''
        Hadj-Hamou longitudinal deformation model
        :param reg_model: Registration model
        :param t0: time 0
        :param t1: time 1
        '''
        super().__init__(t0=t0, t1=t1)
        self.model = model
        self.device = device


    def forward(self, data: tio.SubjectsDataset) -> torch.Tensor:
        denum = 0
        in_shape = data[0]['image'][tio.DATA].shape[1:]
        num = torch.zeros([1, 3] + list(in_shape)).to(self.device)
        transformationPairs = list(itertools.combinations(range(len(data)), 2))
        with torch.no_grad():
            for i, j in transformationPairs:
                sample_i = data[i]
                sample_j = data[j]
                time_ij = sample_j['age'] - sample_i['age']
                velocity_ij = self.reg_model(torch.cat([sample_i['image'][tio.DATA].float().unsqueeze(dim=0).to(self.device),
                                             sample_j['image'][tio.DATA].unsqueeze(dim=0).float().to(self.device)], dim=1))
                num += velocity_ij * time_ij
                denum += time_ij * time_ij
            velocity = num / denum if denum != 0 else torch.zeros_like(num)
        return velocity
