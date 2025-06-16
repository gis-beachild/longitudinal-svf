#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import csv
import os.path
import argparse
import monai
import torch
import torchio as tio
import pytorch_lightning as pl
from monai.metrics import DiceMetric
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from mpmath.identification import transforms
from torch.nn.functional import dropout
from utils.grid_utils import warp, compose
from modules.registration import RegistrationModule
from modules.longitudinal_deformation import OurLongitudinalDeformation
from utils import  create_directory, subjects_from_csv
from modules.blocks.inr import time_encoding

def test(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loggers = pl.loggers.TensorBoardLogger(save_dir= "./log" , name=None)
    save_path = loggers.log_dir.replace('log', "Results")
    create_directory(save_path)
    if args.save_image:
        create_directory(os.path.join(save_path, "images"))
        create_directory(os.path.join(save_path, "flow"))
        create_directory(os.path.join(save_path, "label"))
    ## Config Dataset / Dataloader

    subjects_list = subjects_from_csv(dataset_path=args.csv, lambda_age=lambda x: (x -args.t0) / (args.t1 - args.t0))
    subjects_dataset = tio.SubjectsDataset(subjects_list, transform=None)

    transforms = tio.Compose([
        tio.CropOrPad(target_shape=args.csize),
        tio.Resize(target_shape=args.rsize),
        tio.RescaleIntensity(out_min_max=(0, 1), percentiles=(0.5, 99.5), masking_method='label'),
    ])

    reverse_transform = tio.Compose([
        tio.Resize(target_shape=args.csize),
        tio.CropOrPad(target_shape=subjects_dataset[0]["image"][tio.DATA].shape[1:]),
    ])

    source_subject = None
    target_subject = None
    for s in subjects_dataset:
        if s.age == 0:
            source_subject = s
        if s.age == 1:
            target_subject = s


    reg_model = RegistrationModule(model=monai.networks.nets.AttentionUnet(dropout=0.1,
            spatial_dims=3, in_channels=1, out_channels=3, channels=[8, 16, 32, 64], strides=[2, 2, 2]), int_steps=9)
    # Load models
    model = OurLongitudinalDeformation(
        time_mode=args.time_mode, hidden_dim=args.mlp_hidden_dim, t0=args.t0, t1=args.t1,
        reg_model=reg_model)
    print(model)
    model.load_state_dict(torch.load(args.load), strict=True)
    model.eval().to(device)
    dice_metric = DiceMetric(include_background=True, reduction="none", num_classes=20)
    with open(save_path + "/results.csv", mode='w') as file:
        header = ["time", "mDice", "Cortex", "Ventricule", "all"]
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()
        with torch.no_grad():
            source_subject = transforms(source_subject)
            target_subject_transformed = transforms(target_subject)
            source_image = torch.unsqueeze(source_subject["image"][tio.DATA], 0).to(device)
            source_label = torch.unsqueeze(source_subject["label"][tio.DATA], 0).to(device)
            d, w, h = source_image.shape[2:]
            velocity = model.forward(torch.unsqueeze(source_subject["image"][tio.DATA], 0).to(device).float())
            x  = []
            y = []
            for subject in subjects_dataset:
                age = subject['age'] * (args.t1 - args.t0) + args.t0
                timed_velocity = model.encode_time(torch.Tensor([subject['age']]).to(device)) * velocity
                forward_flow = model.reg_model.velocity2displacement(timed_velocity)
                warped_source_image = warp(source_image.float(), forward_flow)
                warped_source_label = warp(source_label.float(), forward_flow, mode='nearest')
                warped_subject = tio.Subject(
                    image=tio.ScalarImage(tensor=warped_source_image.detach().cpu().squeeze(0), affine=source_subject['image'].affine),
                    label=tio.LabelMap(tensor=warped_source_label.squeeze(0).int().detach().cpu(), affine=source_subject['label'].affine),
                    flow=tio.ScalarImage(tensor=forward_flow.squeeze(0).detach().cpu(), affine=source_subject['image'].affine)
                )
                sub = transforms(subject)
                warped_subject["flow"][tio.DATA] = (torch.tensor(warped_subject["flow"].spacing).view(3, 1, 1, 1) *  warped_subject["flow"][tio.DATA])
                dice = dice_metric(warped_subject['label'][tio.DATA].cpu().unsqueeze(0), sub['label'][tio.DATA].to(device).cpu().unsqueeze(0))
                writer.writerow({
                    'time': age,
                    "mDice": torch.mean(dice[0][1:]).item(),
                    "Cortex": torch.mean(dice[0][3:5]).item(),
                    "Ventricule": torch.mean(dice[0][7:9]).item(),
                    "all": dice[0].cpu().numpy()
                })


                print(age, torch.mean(dice[0][3:5]).item())
                loggers.experiment.add_scalar("Dice ventricule", torch.mean(dice[0][7:9]).item(), age)
                loggers.experiment.add_scalar("Dice cortex", torch.mean(dice[0][3:5]).item(), age)
                loggers.experiment.add_scalar("mDice", torch.mean(dice[0]).item(), age)
                warped_subject['label'][tio.DATA] = torch.argmax(warped_subject['label'][tio.DATA], dim=0).int().unsqueeze(0)
                if args.save_image:
                    warped_subject['image'].save(os.path.join(save_path, "images", str(age) + "_warped_source_image.nii.gz"))
                    warped_subject['label'].save(os.path.join(save_path, "label", str(age) + "_label.nii.gz"))
                    warped_subject['flow'].save(os.path.join(save_path, "flow", str(age) + "_flow.nii.gz"))
                x.append(age)
                y.append(torch.mean(dice[0][3:5]).item())
                
    if model.time_mode == 'mlp':
        model.temp_model.eval()
        x = np.arange(0, 1, 0.06)
        y = np.zeros_like(x)
        for i in range(len(x)):
            y[i] = model.encode_time(torch.Tensor([x[i]]).to(device)).detach().cpu().numpy()
        plt.plot(x, y)
        plt.plot(x, x, '--')
        plt.show()

# %% Main program
if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    parser = argparse.ArgumentParser(description='Beo registration.yaml 3D Longitudinal Images with MLP model')
    parser.add_argument('--csv', type=str, help='Path to the csv file')
    parser.add_argument('--t0', type=int, help='Initial time point', default=21)
    parser.add_argument('--t1', type=int, help='Final time point', default=36)
    parser.add_argument('--load', type=str, help='Path to the model', default='')
    parser.add_argument('--rsize', type=int, nargs='+', help='Resize shape', default=[192, 224, 192])
    parser.add_argument('--csize', type=int, nargs='+', help='Cropsize shape', default=[192, 224, 192])
    parser.add_argument('--num_classes', type=int, help='Number of classes', default=20)
    parser.add_argument('--time_mode', type=str, help='SVF Temporal mode', choices={'mlp', 'linear'}, default='mlp')
    parser.add_argument('--save_image', type=bool, help='Save MRI', default=True)
    parser.add_argument('--mlp_hidden_dim', type=int, help='Hidden size of the MLP model',
                        default=[128, 128, 128, 128]),
    parser.add_argument('--mlp_num_layers', type=int, help='Number layer of the MLP', default=4),
    args = parser.parse_args()
    test(args=args)

