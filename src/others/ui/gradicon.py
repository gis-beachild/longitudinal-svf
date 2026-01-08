import sys
import torch
import argparse
import torchio as tio
from torch import Tensor
import numpy as np
from unigradicon import get_multigradicon, get_unigradicon
import gc
from registration_svf.utils.grid_utils import warp
import torch.nn as nn

class GradIcon(nn.Module):
    def __init__(self, mode='multigradicon'):
        '''
        Multigradicon or Unigradicon registration module
        :param mode: unigradicon or multigradicon model
        '''
        mode = mode.lower()
        if mode == 'multigradicon':
            super().__init__(model=get_multigradicon())
        else:
            super().__init__(model=get_unigradicon())

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        _ = self.model(source, target)
        return self.model.phi_AB_vectorfield

    def forward_backward_flow_registration(self, source: Tensor, target: Tensor):
        _ = self.forward(source, target)
        return self.model.phi_AB_vectorfield, self.model.phi_BA_vectorfield



def main(source_subject, target_subject, mode):
    source_subject_shape = source_subject["image"].shape[1:]
    crop_pad_shape = max(int(np.ceil(max(source_subject_shape) / 32) * 32), 175)
    print("Crop or Pad shape : ", crop_pad_shape)
    transform_resized = tio.Compose(
        [
            # Clamp intensities to the 1st and 99th percentiles
            tio.transforms.CropOrPad(target_shape=crop_pad_shape),
            tio.Resize(target_shape=(175, 175, 175)),
        ]
    )
    normalize = tio.Compose(
        [
            tio.transforms.RescaleIntensity(percentiles=(0.1, 99.9)),
            tio.transforms.Clamp(out_min=0, out_max=1)
        ]
    )


    reversed_transform = tio.Compose(
        [
            tio.Resize(target_shape=crop_pad_shape),
            tio.transforms.CropOrPad(target_shape=source_subject_shape),
        ]
    )

    one_hot = tio.OneHot()
    model = GradIcon(mode=mode)
    model.cuda()
    source_image = source_subject["image"]
    target_image = target_subject["image"]
    source_label = source_subject["label"]
    if source_label[tio.DATA].shape[0] == 1:
        source_label = tio.LabelMap(tensor=source_label[tio.DATA], affine=source_label.affine)
        source_label= one_hot(source_label)
        print("Number of classes : ", source_label[tio.DATA].shape[0])

    source_image = transform_resized(source_image)
    target_image = transform_resized(target_image)
    source_label = transform_resized(source_label)
    source_img_norm = normalize(source_image)
    target_img_norm = normalize(target_image)

    forward_flow, backward_flow = model.forward_backward_flow_registration(source_img_norm[tio.DATA].float().unsqueeze(0).cuda(), target_img_norm[tio.DATA].float().unsqueeze(0).cuda())
    u = (model.model.phi_AB_vectorfield.detach() - model.model.identity_map.detach()).squeeze()
    _, D, W, H = u.shape

    u_norm = u.cpu() * torch.tensor([D - 1, W - 1, H - 1]).view(3, 1, 1, 1).cpu()
    warped_source = warp(source_image[tio.DATA].float().cuda().unsqueeze(0), u_norm.cuda().unsqueeze(0))
    print(source_label[tio.DATA].float().cuda().unsqueeze(0).shape)
    warped_label_source = warp(source_label[tio.DATA].float().cuda().unsqueeze(0), u_norm.cuda().unsqueeze(0))
    print("Number of classes : ", warped_label_source.shape[1])
    warped_label_source = torch.argmax(warped_label_source, dim=1)
    torch.cuda.empty_cache()
    warped_source = tio.ScalarImage(tensor=warped_source.squeeze(dim=0).cpu().detach(),
                        affine=source_img_norm.affine)
    warped_label_source = tio.LabelMap(tensor=warped_label_source.int().cpu(),
                     affine=source_img_norm.affine)
    phi_AB = tio.ScalarImage(tensor=(u_norm).cpu(), affine=source_img_norm.affine)

    warped_source = reversed_transform(warped_source)
    warped_label_source = reversed_transform(warped_label_source)
    phi_AB = reversed_transform(phi_AB)
    spacing = torch.Tensor(source_subject['image'].spacing).view(3, 1, 1, 1).cpu()
    phi_AB[tio.DATA] = phi_AB[tio.DATA] * spacing
    model.model.clean()
    gc.collect()

    return warped_source, warped_label_source, phi_AB


