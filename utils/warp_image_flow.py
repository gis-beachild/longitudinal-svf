import argparse
import monai
import torch
import torchio as tio

if __name__ == '__main__':

    argparse = argparse.ArgumentParser("Image warping from a deformation field")
    argparse.add_argument('--i', help="Image to deform", type=str, required=True)
    argparse.add_argument('--f', help="Deformation file path", type=str, required=True)
    argparse.add_argument('--o', help="Output file", type=str, required=False, default='./output.nii.gz')
    argparse.add_argument('--m', help="mm format to voxel", type=float, required=False, default=[1.0,1.0,1.0])
    args = argparse.parse_args()

    Warp = monai.networks.blocks.warp.Warp(padding_mode='zeros', mode='bilinear', jitter=False)
    source = tio.ScalarImage(args.i)
    flow = tio.ScalarImage(args.f)
    warp_image = Warp(source[tio.DATA].unsqueeze(0).double(),  (flow[tio.DATA].double() / torch.tensor(args.m).view(-1, 1, 1, 1)).unsqueeze(0))

    tio.ScalarImage(tensor=warp_image.squeeze(0).cpu().detach(), affine=source.affine).save(args.o)
