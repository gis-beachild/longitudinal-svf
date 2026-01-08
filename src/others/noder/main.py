import train
import argparse

import yaml
import torch
import os
import numpy as np
from train import load_imgs_and_time
from Network.DynamicNet import DynamicNet
from registration_svf.utils.grid_utils import warp
from Utils.Utls import generate_grid3D_tensor
from torchdiffeq import odeint_adjoint as odeint
import datetime
import torchio as tio
import pandas as pd

def main(args):
    with open(args.dataset_yaml, "r") as f:
        config = yaml.safe_load(f)
    csize = config['csize']
    name = config['name']
    num_classes = config['num_classes']
    date_format = config['date_format']
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    subject_path = name
    result_save_path = f"{args.savePath}/noder/{subject_path}/"
    result_save_path_images = os.path.join(result_save_path, "images")
    result_save_path_flows = os.path.join(result_save_path, "flows")
    result_save_path_seg = os.path.join(result_save_path, "parcellations")
    os.makedirs(result_save_path_images, exist_ok=True)
    os.makedirs(result_save_path_seg, exist_ok=True)
    os.makedirs(result_save_path_flows, exist_ok=True)

    imgs, seg, times = load_imgs_and_time(config['csv_path'], padding=csize, date_format=date_format)
    test_List = imgs
    test_times = times
    seq_length = len(test_List)
    print(len(test_times))
    im_shape = test_List[0].shape

    Network = DynamicNet(img_sz=im_shape,
                         smoothing_kernel='AK',
                         smoothing_win=15,
                         smoothing_pass=1,
                         ds=2,
                         bs=32
                         ).to(device)

    Network.load_state_dict(torch.load(os.path.join(savePath, "model.pkl")))
    Network.eval()
    # %%
    scale_factor = torch.tensor(im_shape).to(device).view(1, 3, 1, 1, 1) * 1.
    grid = generate_grid3D_tensor(im_shape).unsqueeze(0).to(device)  # [-1,1] 1*3*144*176*144

    df = pd.read_csv(config["csv_path"])
    lst_data = []
    for index, row in df.iterrows():
        lst_data.append((row['age'], row['image'], row['label']))
    if date_format is not None:
        lst_data = sorted(lst_data, key=lambda x: datetime.datetime.strptime(x[0], date_format))
    else:
        lst_data = sorted(lst_data, key=lambda x: x[0])
    image_template = tio.ScalarImage(lst_data[0][1])
    label_template = tio.ScalarImage(lst_data[0][2])
    all_phi = odeint(func=Network, y0=grid, t=torch.tensor(times).to(device), method="rk4", rtol=1e-3, atol=1e-5).to(
        device)
    all_phi = (all_phi + 1.) / 2. * scale_factor  # [-1, 1] -> voxel spacing
    grid_voxel = (grid + 1.) / 2. * scale_factor  # [-1, 1] -> voxel spacing

    reverse_crop = tio.CropOrPad(image_template.shape[1:])
    img_shape = (all_phi[0].shape[2], all_phi[0].shape[3], all_phi[0].shape[4])
    print(img_shape)
    crop = tio.CropOrPad(img_shape)
    image_template = crop(image_template)
    label_template = crop(label_template)
    if label_template[tio.DATA].shape[0] == 1:
        label_template = tio.LabelMap(tensor=label_template[tio.DATA].int(), affine=label_template.affine)
        one_hot = tio.OneHot()
        label_template = one_hot(label_template)
    for n in range(0, seq_length):
        phi = all_phi[n]
        df = phi - grid_voxel  # with grid -> without grid
        warped_moving = warp(image_template[tio.DATA].unsqueeze(0).cuda().float(), df)
        warped_moving = warped_moving.squeeze(0)
        reverse_crop(tio.ScalarImage(tensor=warped_moving.detach().cpu(), affine=image_template.affine)).save(
            '%s/warped-t%d.nii.gz' % (result_save_path_images, n))
        (reverse_crop(
            tio.ScalarImage(tensor=df.squeeze().detach().cpu() * torch.Tensor(image_template.spacing).view(3, 1, 1, 1),
                            affine=image_template.affine))).save(
            '%s/df-t%d.nii.gz' % (result_save_path_flows, n))

        warped_label_moving = warp(label_template[tio.DATA].unsqueeze(0).cuda().float(), df)
        reverse_crop(
            tio.LabelMap(tensor=torch.argmax(warped_label_moving.detach().cpu(), dim=1), affine=label_template.affine)).save(
            '%s/warped-t%d_label.nii.gz' % (result_save_path_seg, n))
        print(result_save_path_seg)


if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--dataset_yaml', type=str, help='Subject name', default="/home/florian/Documents/Programs/longitudinal-svf-pair/src/ours-svf/svf-pair-long/configs/data/gholipour.yaml")
    argparse.add_argument('--savePath', type=str, default="./model-save", help='Model save path')
    argparse.add_argument('--lambdaGrad', type=float, default=0.005, help='Lambda for gradient loss')
    argparse.add_argument('--epochs', type=int, default=3000, help='Number of epochs')
    argparse.add_argument('--train', type=bool, default=False, help='Training before inference')
    args = argparse.parse_args()
    if args.train:
        train.main(args)
    main(args)