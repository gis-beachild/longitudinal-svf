import argparse
import monai
import torch
import torchio as tio
import numpy as np
if __name__ == '__main__':

    argparse = argparse.ArgumentParser("Image warping from a deformation field")
    argparse.add_argument('--i', help="Image to deform", type=str, required=False, default='/home/florian/Documents/Dataset/dHCP/structural/tissue-t21.00_dhcp-19.nii.gz')
    argparse.add_argument('--f', help="Deformation file path", type=str, required=False, default='/home/florian/Documents/Programs/longitudinal-svf/src/outputs/longitudinal/2025-06-12_17-02-30/forward_dvf.nii.gz')
    argparse.add_argument('--o', help="Output file", type=str, required=False, default='./output.nii.gz')
    argparse.add_argument('--m', help="mm format to voxel", type=float, required=False, default=[.5,.5,.5])
    args = argparse.parse_args()
    # Load the source image (moving image) and flow

    # Load source image and deformation field
    source = tio.ScalarImage(args.i)
    transform = tio.Compose([
        tio.CropOrPad(221),
        tio.Resize(128)
    ])
    source = transform(source)
    flow = tio.ScalarImage(args.f)

    # Extract data tensors
    source_tensor = source[tio.DATA]  # (1, Z, Y, X)
    flow_tensor = flow[tio.DATA]      # (3, Z, Y, X)

    # Get voxel spacing from source image (x, y, z)
    spacing = torch.tensor(source.spacing).view(3, 1, 1, 1)  # torch (3,1,1,1)

    # Convert flow from mm to voxel units
    voxel_flow = flow_tensor / spacing

    # Flip Y axis (P to A) for LAS convention
    voxel_flow[1] *= -1  # Y-axis inversion (Posterior <-> Anterior)

    # Warp the source image using MONAI
    warp = monai.networks.blocks.warp.Warp(padding_mode='zeros', mode='nearest')
    warped_image = warp(source_tensor.unsqueeze(0).double(), voxel_flow.unsqueeze(0).double())

    # Save warped image
    tio.ScalarImage(tensor=warped_image.squeeze(0).cpu(), affine=source.affine).save(args.o)