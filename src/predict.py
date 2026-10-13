"""Unified inference CLI for pairwise and longitudinal (linear/mlp) SVF registration.

Author: Fl0rian
"""
import os.path
import argparse
from typing import Union

import nibabel as nib
import numpy as np
import torch
import torchio as tio
import yaml
from monai.metrics import DiceMetric # type: ignore
from data.longitudinal_datamodule import LongitudinalDataModule
from data.pairwise_datamodule import PairwiseRegistrationDataModule
from modules.svf_registration import SVFRegistrationModule
from modules.unet import DyNUnet
from utils.grid_utils import warp

from modules.longitudinal_model import LongitudinalDeformation

# CLI --mode value -> LongitudinalDeformation.time_mode value
MODE_TO_TIME_MODE = {
    "longitudinal_linear": "linear",
    "longitudinal_mlp": "mlp",
}


def _datamodule_kwargs(config: dict) -> dict:
    """Translate a current data YAML into arguments shared by both datamodules."""
    kwargs = {
        'data_dir': config.get('csv_path', ''),
        'batch_size': 1,
        'num_workers': 0,
        'rsize': config['rsize'],
        'csize': config['csize'],
        'num_classes': config['num_classes'],
    }
    if config.get('train_json') is not None:
        kwargs.update(
            root_dir=config['root_dir'],
            train_json=config['train_json'],
            val_json=config.get('val_json'),
            merge_labels_0_1=config.get('merge_labels_0_1', False),
        )
    return kwargs


def _load_weights(module: torch.nn.Module, path: str, device: Union[str, torch.device]) -> None:
    """Load an inference state dictionary on the selected device."""
    module.load_state_dict(torch.load(path, map_location=device, weights_only=True))


def _flow_to_ras_mm(flow_voxel: torch.Tensor, affine: np.ndarray) -> torch.Tensor:
    """Convert ``(3, I, J, K)`` voxel-axis displacements to RAS millimetres."""
    linear = torch.as_tensor(
        np.asarray(affine[:3, :3]), dtype=flow_voxel.dtype, device=flow_voxel.device
    )
    return torch.einsum("rc,cijk->rijk", linear, flow_voxel)


def _save_displacement_nifti(
        flow_ras_mm: torch.Tensor, affine: np.ndarray, output_path: str) -> None:
    """Save RAS-mm vectors as a NIfTI ``DISPVECT`` deformation field."""
    if flow_ras_mm.ndim != 4 or flow_ras_mm.shape[0] != 3:
        raise ValueError(
            f"flow must have shape (3, I, J, K), got {tuple(flow_ras_mm.shape)}"
        )
    data = flow_ras_mm.detach().cpu().permute(1, 2, 3, 0).unsqueeze(3).numpy()
    image = nib.Nifti1Image(data.astype(np.float32, copy=False), affine)
    image.header.set_intent(1006, name="displacement")
    image.header.set_xyzt_units(xyz="mm")
    nib.save(image, output_path)


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


def run_pairwise(source_subject: tio.Subject, target_subject: tio.Subject,
                 transform: tio.transforms.Compose, model: SVFRegistrationModule,
                 device: Union[str, torch.device], csize,
                 savePath: str = './', index: int = 0) -> None:
    """Register one source/target pair with the datamodule's inference transform."""
    result_save_path_images = os.path.join(savePath, "images")
    result_save_path_flows = os.path.join(savePath, "flows")
    result_save_path_seg = os.path.join(savePath, "parcellations")
    os.makedirs(result_save_path_images, exist_ok=True)
    os.makedirs(result_save_path_seg, exist_ok=True)
    os.makedirs(result_save_path_flows, exist_ok=True)

    source_shape = source_subject.image.data.shape[1:]
    reverse_transform = tio.transforms.Compose([
        tio.transforms.Resize(target_shape=csize),
        tio.transforms.CropOrPad(target_shape=source_shape),
    ])

    source_subject_transformed = transform(source_subject)
    target_subject_transformed = transform(target_subject)

    forward_flow, _ = inference_pairwise(source_subject_transformed, target_subject_transformed, model, device)

    source_img = source_subject_transformed.image.data.unsqueeze(0).to(device).float()
    source_label = source_subject_transformed.label.data.unsqueeze(0).to(device).float()
    warped_source_img = warp(source_img, forward_flow)
    warped_source_label = warp(source_label, forward_flow)

    flow_ras_mm = _flow_to_ras_mm(
        forward_flow.squeeze(0), source_subject_transformed.image.affine
    )
    warped_subjects = tio.Subject(
        image=tio.ScalarImage(tensor=warped_source_img.squeeze(0).cpu().detach(), affine=source_subject_transformed.image.affine),
        label=tio.LabelMap(tensor=torch.argmax(warped_source_label, dim=1).cpu().detach(), affine=source_subject_transformed.label.affine),
        flow=tio.ScalarImage(tensor=flow_ras_mm.detach().cpu(), affine=source_subject_transformed.image.affine)
    )

    warped_subjects = reverse_transform(warped_subjects)
    warped_subjects.image.save(os.path.join(result_save_path_images, f"warped-t{index}.nii.gz"))
    _save_displacement_nifti(
        warped_subjects.flow.data,
        warped_subjects.flow.affine,
        os.path.join(result_save_path_flows, f"df-t{index}.nii.gz"),
    )
    warped_subjects.label.save(os.path.join(result_save_path_seg, f"warped-t{index}_label.nii.gz"))


def predict_pairwise(args: argparse.Namespace) -> None:
    """Run pairwise inference of the ``t0`` reference subject against every other subject in the dataset."""
    with open(args.dataset_yaml, "r") as f:
        config = yaml.safe_load(f)
    datamodule = PairwiseRegistrationDataModule(**_datamodule_kwargs(config))
    datamodule.setup('predict')
    json_data = config.get('train_json') is not None
    if json_data:
        source = next(subject for subject in datamodule.subjects
                      if subject['age'] == float(config['t0']))
        targets = [subject for subject in datamodule.val_subjects
                   if subject['age'] != float(config['t0'])]
    else:
        source_index = int(config['t0'])
        source = datamodule.subjects[source_index]
        targets = list(datamodule.val_subjects)
        del targets[source_index]

    transform = datamodule.inference_transform()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_svf_model(device)
    _load_weights(model, args.load, device)
    output_name = args.name or os.path.join(config['name'], 'pairwise')
    output_path = os.path.join(args.savePath, output_name)
    os.makedirs(output_path, exist_ok=True)
    for i, target in enumerate(targets):
        print(f"Processing {i + 1} / {len(targets)}: {target.image.path}")
        run_pairwise(source, target, transform, model, device, config['csize'],
                     savePath=output_path, index=i)


def predict_longitudinal(args: argparse.Namespace) -> None:
    """Run longitudinal inference: warp the ``t0`` subject to every subject's age and report per-timepoint Dice."""
    time_mode = MODE_TO_TIME_MODE[args.mode]
    with open(args.dataset_yaml, "r") as f:
        config = yaml.safe_load(f)
    t0 = config['t0']
    t1 = config.get('tn', config.get('t1'))
    name = config['name']
    result_save_path = os.path.join(args.savePath, name, args.mode)
    result_save_path_images = os.path.join(result_save_path, "images")
    result_save_path_flows = os.path.join(result_save_path, "flows")
    result_save_path_seg = os.path.join(result_save_path, "parcellations")
    os.makedirs(result_save_path_images, exist_ok=True)
    os.makedirs(result_save_path_seg, exist_ok=True)
    os.makedirs(result_save_path_flows, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    datamodule = LongitudinalDataModule(
        **_datamodule_kwargs(config),
        t0=t0,
        t1=t1,
        date_format=config.get('date_format'),
    )
    datamodule.setup('predict')
    subjects = datamodule.val_subjects
    source_subject = next(subject for subject in datamodule.train_subjects if subject['age'] == 0)
    target_subject = next(subject for subject in datamodule.train_subjects if subject['age'] == 1)
    transform = datamodule.inference_transform()

    source_shape = source_subject.image.data.shape[1:]
    reverse_transform = tio.transforms.Compose([
        tio.transforms.Resize(target_shape=config['csize']),
        tio.transforms.CropOrPad(target_shape=source_shape, padding_mode='reflect'),
    ])

    source_processed = transform(source_subject)
    target_processed = transform(target_subject)
    source_input = source_processed.image.data.unsqueeze(0).float().to(device)
    target_input = target_processed.image.data.unsqueeze(0).float().to(device)
    input_tensor = torch.cat([source_input, target_input], dim=1)

    svf_model = build_svf_model(device)
    model = LongitudinalDeformation(svf_model=svf_model, time_mode=time_mode, t0=t0, t1=t1)
    _load_weights(model.svf_model, args.load, device)
    if time_mode == 'mlp' and args.load_temporal:
        _load_weights(model.mlp_model, args.load_temporal, device)
    model.eval().to(device)

    with torch.no_grad():
        velocity = model.forward(input_tensor)
        source_image = source_processed.image.data.unsqueeze(0).float().to(device)
        source_label = source_processed.label.data.unsqueeze(0).float().to(device)
        for i, subject in enumerate(subjects):
            subject_processed = transform(subject)
            time = torch.tensor([subject['age']], dtype=velocity.dtype, device=device)
            timed_velocity = model.encode_time(time) * velocity
            forward_flow = model.svf_model.velocity2displacement(timed_velocity)
            warped_source_image = warp(source_image, forward_flow)
            warped_source_scores = warp(source_label, forward_flow)
            warped_source_label = torch.argmax(warped_source_scores, dim=1, keepdim=True)
            warped_source_one_hot = torch.nn.functional.one_hot(
                warped_source_label.squeeze(1), num_classes=config['num_classes']
            ).movedim(-1, 1).float()
            j = format_number(i, len(subjects))

            flow_ras_mm = _flow_to_ras_mm(
                forward_flow.squeeze(0), source_processed.image.affine
            )
            warped_subject = tio.Subject(
                image=tio.ScalarImage(tensor=warped_source_image.detach().cpu().squeeze(0), affine=source_processed.image.affine),
                label=tio.LabelMap(tensor=warped_source_label.squeeze(0).int().detach().cpu(), affine=source_processed.label.affine),
                flow=tio.ScalarImage(tensor=flow_ras_mm.detach().cpu(), affine=source_processed.image.affine)
            )
            dice_metric = DiceMetric(include_background=True)
            dice = dice_metric(
                warped_source_one_hot,
                subject_processed.label.data.unsqueeze(0).to(device),
            )
            print(dice)
            dice_metric.reset()
            warped_subject = reverse_transform(warped_subject)
            warped_subject.image.save(os.path.join(result_save_path_images, f"warped-t{j}.nii.gz"))
            _save_displacement_nifti(
                warped_subject.flow.data,
                warped_subject.flow.affine,
                os.path.join(result_save_path_flows, f"df-t{j}.nii.gz"),
            )
            warped_subject.label.save(os.path.join(result_save_path_seg, f"warped-t{j}_label.nii.gz"))


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    parser = argparse.ArgumentParser(description='Inference for pairwise/longitudinal SVF registration')
    parser.add_argument('--mode', type=str, required=False, choices=['pairwise', 'longitudinal_linear', 'longitudinal_mlp'], default='pairwise', help='Mode of inference: pairwise or longitudinal (linear/mlp)')
    parser.add_argument('--dataset_yaml', type=str, required=True, help='Path to the dataset yaml file')
    parser.add_argument('--savePath', type=str, required=True, help='Path to the output directory')
    parser.add_argument('--load', type=str, required=True, help='Path to the SVF model weights (model.pth)')
    parser.add_argument('--load_temporal', type=str, default=None,
                        help='Path to the temporal MLP weights (temporal_model.pth), only used with --mode longitudinal_mlp')
    parser.add_argument('--name', type=str, default=None,
                        help='Optional pairwise output subdirectory (defaults to <dataset>/pairwise)')
    args = parser.parse_args()

    if args.mode == 'pairwise':
        predict_pairwise(args)
    else:
        if args.mode == 'longitudinal_mlp' and args.load_temporal is None:
            parser.error('--load_temporal is required when --mode longitudinal_mlp')
        predict_longitudinal(args)
