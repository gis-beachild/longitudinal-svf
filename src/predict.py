"""Unified inference CLI for pairwise and longitudinal (linear/mlp) SVF registration.

Author: Fl0rian
"""
import os.path
import argparse
from datetime import datetime
from typing import Union

import torch
import torchio as tio
import pandas as pd
import yaml
from monai.metrics import DiceMetric # type: ignore
from modules.svf_registration import SVFRegistrationModule
from modules.unet import DyNUnet
from utils.grid_utils import warp

from modules.longitudinal_model import LongitudinalDeformation

# CLI --mode value -> LongitudinalDeformation.time_mode value
MODE_TO_TIME_MODE = {
    "longitudinal_linear": "linear",
    "longitudinal_mlp": "mlp",
}


def format_number(n: int, max_n: int) -> str:
    """
    Format a number with leading zeros based on the maximum number.
    """
    max_digits = len(str(max_n))
    return str(n).zfill(max_digits)


def build_svf_model(device: Union[str, torch.device]) -> SVFRegistrationModule:
    """Build the fixed DyNUnet-backed SVF registration model used for inference."""
    return SVFRegistrationModule(
        model=DyNUnet(
            in_channels=2,
            out_channels=3,
            kernel_size=[[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
            strides=[[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]),
        int_steps=9).eval().to(device)


def inference_pairwise(source: tio.Subject, target: tio.Subject, model: SVFRegistrationModule, device: Union[str, torch.device]):
    """Predict forward/backward displacement fields between ``source`` and ``target`` subjects."""
    model.eval().to(device)
    source_img = source.image.data.unsqueeze(0).to(device)
    target_img = target.image.data.unsqueeze(0).to(device)
    with torch.no_grad():
        velocity = model(torch.cat([source_img, target_img], dim=1))
    forward_flow = model.velocity2displacement(velocity)
    backward_flow = model.velocity2displacement(-velocity)
    return forward_flow, backward_flow


def run_pairwise(img_src: str, img_target: str, lbl_src: str, lbl_target: str, csize, rsize, num_classes: int,
                  load: str, savePath: str = './', index: int = 0) -> None:
    """Register ``img_src``/``lbl_src`` onto ``img_target``/``lbl_target`` with a pairwise SVF model and save the outputs."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    result_save_path_images = os.path.join(savePath, "images")
    result_save_path_flows = os.path.join(savePath, "flows")
    result_save_path_seg = os.path.join(savePath, "parcellations")
    os.makedirs(result_save_path_images, exist_ok=True)
    os.makedirs(result_save_path_seg, exist_ok=True)
    os.makedirs(result_save_path_flows, exist_ok=True)

    source_subject = tio.Subject(
        image=tio.ScalarImage(img_src),
        label=tio.LabelMap(lbl_src),
    )
    target_subject = tio.Subject(
        image=tio.ScalarImage(img_target),
        label=tio.LabelMap(lbl_target),
    )

    transforms_input = tio.transforms.Compose([
        tio.transforms.CropOrPad(target_shape=csize),
        tio.transforms.Resize(target_shape=rsize),
        tio.transforms.RescaleIntensity(out_min_max=(0, 1), percentiles=(0.5, 99.5), masking_method='label'),
        tio.transforms.Clamp(out_min=0, out_max=1, include=['image'])
    ])
    transforms_without_norm = tio.transforms.Compose([
        tio.transforms.CropOrPad(target_shape=csize),
        tio.transforms.Resize(target_shape=rsize),
        tio.transforms.OneHot()
    ])
    reverse_transform = tio.transforms.Compose([
        tio.transforms.Resize(target_shape=csize),
        tio.transforms.CropOrPad(target_shape=source_subject.image.data.shape[1:]),
    ])

    source_subject_transformed = transforms_input(source_subject)
    target_subject_transformed = transforms_input(target_subject)
    source_subject_warp_transformed = transforms_without_norm(source_subject)

    model = build_svf_model(device)
    model.load_state_dict(torch.load(load))

    forward_flow, backward_flow = inference_pairwise(source_subject_transformed, target_subject_transformed, model, device)

    source_img = source_subject_warp_transformed.image.data.unsqueeze(0).to(device).float()
    source_label = source_subject_warp_transformed.label.data.unsqueeze(0).to(device).float()
    warped_source_img = warp(source_img, forward_flow)
    warped_source_label = warp(source_label, forward_flow)

    warped_subjects = tio.Subject(
        image=tio.ScalarImage(tensor=warped_source_img.squeeze(0).cpu().detach(), affine=source_subject_transformed.image.affine),
        label=tio.ScalarImage(tensor=torch.argmax(warped_source_label, dim=1).cpu().detach(), affine=source_subject_transformed.label.affine),
        flow=tio.ScalarImage(tensor=forward_flow.squeeze(0).detach().cpu(), affine=source_subject_transformed.image.affine)
    )

    warped_subjects = reverse_transform(warped_subjects)
    warped_subjects.flow.data = warped_subjects.flow.data * torch.tensor(source_subject.spacing).view(3, 1, 1, 1)
    warped_subjects.image.save(os.path.join(result_save_path_images, f"warped-t{index}.nii.gz"))
    warped_subjects.flow.save(os.path.join(result_save_path_flows, f"df-t{index}.nii.gz"))
    warped_subjects.label.save(os.path.join(result_save_path_seg, f"warped-t{index}_label.nii.gz"))


def predict_pairwise(args: argparse.Namespace) -> None:
    """Run pairwise inference of the ``t0`` reference subject against every other subject in the dataset."""
    with open(args.dataset_yaml, "r") as f:
        config = yaml.safe_load(f)
    rsize = config['rsize']
    csize = config['csize']
    num_classes = config['num_classes']
    df = pd.read_csv(config['csv_path'])
    t0 = int(config['t0'])
    lst_images, lst_labels = [], []
    for _, row in df.iterrows():
        lst_images.append(row['image'])
        lst_labels.append(row['label'])
    image_source = lst_images[t0]
    label_source = lst_labels[t0]
    lst_labels = lst_labels[:t0] + lst_labels[t0 + 1:]
    lst_images = lst_images[:t0] + lst_images[t0 + 1:]
    output_path = os.path.join(args.savePath, args.name)
    os.makedirs(output_path, exist_ok=True)
    for i in range(len(lst_images)):
        print(f"Processing {i + 1} / {len(lst_images)} : Image: {lst_images[i]}, Label: {lst_labels[i]}")
        run_pairwise(image_source, lst_images[i], label_source, lst_labels[i], csize, rsize, num_classes, args.load,
                     savePath=output_path, index=i)


def predict_longitudinal(args: argparse.Namespace) -> None:
    """Run longitudinal inference: warp the ``t0`` subject to every subject's age and report per-timepoint Dice."""
    time_mode = MODE_TO_TIME_MODE[args.mode]
    with open(args.dataset_yaml, "r") as f:
        config = yaml.safe_load(f)
    rsize = config['rsize']
    csize = config['csize']
    t0 = config['t0']
    t1 = config['t1']
    csv_path = config['csv_path']
    name = config['name']
    date_format = config['date_format']
    result_save_path = os.path.join(args.savePath, name, args.mode)
    result_save_path_images = os.path.join(result_save_path, "images")
    result_save_path_flows = os.path.join(result_save_path, "flows")
    result_save_path_seg = os.path.join(result_save_path, "parcellations")
    os.makedirs(result_save_path_images, exist_ok=True)
    os.makedirs(result_save_path_seg, exist_ok=True)
    os.makedirs(result_save_path_flows, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    subjects_list = []
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        subject = tio.Subject(
            image=tio.ScalarImage(row['image']),
            label=tio.LabelMap(row['label']),
            age=str(row['age'])
        )
        subjects_list.append(subject)
    subjects_dataset = tio.SubjectsDataset(subjects_list, transform=None)

    transforms_input = tio.transforms.Compose([
        tio.transforms.CropOrPad(csize),
        tio.transforms.Resize(rsize),
        tio.transforms.RescaleIntensity(out_min_max=(0, 1), percentiles=(0.5, 99.5), masking_method='label'),
        tio.transforms.Clamp(out_min=0, out_max=1, include=['image'])
    ])
    transforms_without_norm = tio.transforms.Compose([
        tio.transforms.CropOrPad(target_shape=csize)
    ])
    reverse_transform = tio.transforms.Compose([
        tio.transforms.CropOrPad(target_shape=subjects_dataset[0].image.data.shape[1:], padding_mode='reflect')
    ])

    source_subject = subjects_dataset[t0]
    target_subject = subjects_dataset[t1]

    source_input = transforms_input(source_subject).image.data.unsqueeze(0).to(device)
    target_input = transforms_input(target_subject).image.data.unsqueeze(0).to(device)
    input_tensor = torch.cat([source_input, target_input], dim=1)

    reference_date_t0 = datetime.strptime(source_subject.age, date_format).timestamp() if date_format else float(source_subject.age)
    reference_date_t1 = datetime.strptime(target_subject.age, date_format).timestamp() if date_format else float(target_subject.age)

    svf_model = build_svf_model(device)
    model = LongitudinalDeformation(svf_model=svf_model, time_mode=time_mode, t0=t0, t1=t1)
    model.load_reg_model(args.load)
    if time_mode == 'mlp' and args.load_temporal:
        model.load_temporal(args.load_temporal)
    model.eval().to(device)

    with torch.no_grad():
        velocity = model.forward(input_tensor)
        source_t0 = transforms_without_norm(source_subject)
        source_image = source_t0.image.data.unsqueeze(0).to(device)
        if source_t0.label.data.shape[0] == 1:
            source_t0 = tio.transforms.OneHot()(source_t0)
        source_label = source_t0.label.data.unsqueeze(0).to(device)
        for i in range(len(subjects_dataset)):
            subject = subjects_dataset[i]
            age = datetime.strptime(subject.age, date_format).timestamp() if date_format else float(subject.age)
            age = (age - reference_date_t0) / (reference_date_t1 - reference_date_t0)
            timed_velocity = model.encode_time(torch.Tensor([age]).to(device)) * velocity
            forward_flow = model.svf_model.velocity2displacement(timed_velocity)
            warped_source_image = warp(source_image.float(), forward_flow)
            warped_source_label = torch.argmax(warp(source_label.float(), forward_flow), dim=1).unsqueeze(0)
            j = format_number(i, len(subjects_dataset))
            if subject.label.data.shape[0] == 1:
                target = tio.LabelMap(tensor=subject.label.data.to(device), affine=subject.label.affine)
            else:
                target = tio.LabelMap(tensor=torch.argmax(subject.label.data.unsqueeze(0).to(device), dim=1),
                                      affine=subject.label.affine)

            warped_subject = tio.Subject(
                image=tio.ScalarImage(tensor=warped_source_image.detach().cpu().squeeze(0), affine=source_t0.image.affine),
                label=tio.LabelMap(tensor=warped_source_label.squeeze(0).int().detach().cpu(), affine=source_t0.label.affine),
                flow=tio.ScalarImage(tensor=forward_flow.squeeze(0).detach().cpu(), affine=source_t0.image.affine)
            )
            warped_subject = reverse_transform(warped_subject)
            dice_metric = DiceMetric(include_background=True)
            dice = dice_metric(tio.transforms.OneHot()(warped_subject.label).data.unsqueeze(0).cpu(),
                               tio.transforms.OneHot()(target).data.unsqueeze(0).cpu())
            print(dice)
            dice_metric.reset()
            warped_subject.flow.data = warped_subject.flow.data * torch.tensor(source_subject.spacing).view(3, 1, 1, 1)
            warped_subject.image.save(os.path.join(result_save_path_images, f"warped-t{j}.nii.gz"))
            warped_subject.flow.save(os.path.join(result_save_path_flows, f"df-t{j}.nii.gz"))
            warped_subject.label.save(os.path.join(result_save_path_seg, f"warped-t{j}_label.nii.gz"))


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    parser = argparse.ArgumentParser(description='Inference for pairwise/longitudinal SVF registration')
    parser.add_argument('--mode', type=str, required=True, choices=['pairwise', 'longitudinal_linear', 'longitudinal_mlp'],
                        help='Which trained model to run inference with')
    parser.add_argument('--dataset_yaml', type=str, required=True, help='Path to the dataset yaml file')
    parser.add_argument('--savePath', type=str, required=True, help='Path to the output directory')
    parser.add_argument('--load', type=str, required=True, help='Path to the SVF model weights (model.pth)')
    parser.add_argument('--load_temporal', type=str, default=None,
                        help='Path to the temporal MLP weights (temporal_model.pth), only used with --mode longitudinal_mlp')
    parser.add_argument('--name', type=str, default=None,
                        help='Output subdirectory name, required with --mode pairwise')
    args = parser.parse_args()

    if args.mode == 'pairwise':
        if args.name is None:
            parser.error('--name is required when --mode pairwise')
        predict_pairwise(args)
    else:
        predict_longitudinal(args)
