import argparse
import yaml
import torch
import os
from registration_svf.utils.grid_utils import warp
import datetime
import torchio as tio
import pandas as pd
from registration_svf.registration import RegistrationModule
from hadj_hamou import HadjHamouSVF
import monai


def main(args):
    with open(args.dataset_yaml, "r") as f:
        config = yaml.safe_load(f)
    rsize = config['rsize']
    csize = config['csize']
    t0 = config['t0']
    t1 = config['t1']
    name = config['name']
    num_classes = config['num_classes']
    csv_path = config['csv_path']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    result_save_path = f"./result-save/{name}/"
    result_save_path_images = os.path.join(result_save_path, "parcellations")
    result_save_path_flows = os.path.join(result_save_path, "flows")
    result_save_path_seg = os.path.join(result_save_path, "parcellations")
    os.makedirs(result_save_path_images, exist_ok=True)
    os.makedirs(result_save_path_seg, exist_ok=True)
    os.makedirs(result_save_path_flows, exist_ok=True)

    ## Config Dataset / Dataloader
    subjects_list = []
    df = pd.read_csv(csv_path)
    for index, row in df.iterrows():
        subject = tio.Subject(
            image=tio.ScalarImage(row['image']),
            label=tio.LabelMap(row['label']),
            age=str(row['age'])
        )
        subjects_list.append(subject)

    subjects_dataset = tio.SubjectsDataset(subjects_list, transform=None)

    transforms_input = tio.Compose([
        tio.CropOrPad(target_shape=csize),
        tio.Resize(target_shape=rsize),
        tio.RescaleIntensity(out_min_max=(0, 1), percentiles=(0.5, 99.5), include=['image'])
    ])

    transforms_without_norm = tio.Compose([
        tio.CropOrPad(target_shape=csize),
        tio.Resize(target_shape=rsize),

    ])

    reverse_transform = tio.Compose([
        tio.Resize(target_shape=csize),
        tio.CropOrPad(target_shape=subjects_dataset[0]["image"][tio.DATA].shape[1:]),
    ])

    source_subject = subjects_list[t0]
    target_subject = subjects_list[t1]

    reference_date_t0 = datetime.strptime(source_subject['age'], '%Y/%m/%d') if args.df else int(source_subject['age'])
    reference_date_t1 = datetime.strptime(target_subject['age'], '%Y/%m/%d') if args.df else int(target_subject['age'])

    for i in range(len(subjects_list)):
        subject = subjects_list[i]
        age = datetime.strptime(subject['age'], '%Y/%m/%d') if args.df else int(subject['age'])
        subjects_list[i]['age'] = (age - reference_date_t0) / (reference_date_t1 - reference_date_t0)

    subjects_dataset = tio.SubjectsDataset(subjects_list, transform=None)
    subjects_dataset_transform = tio.SubjectsDataset(subjects_list, transform=transforms_input)
    source_subject = subjects_dataset[t0]
    target_subject = subjects_dataset[t1]
    model = RegistrationModule(
        model=monai.networks.nets.DyNUnet(
            in_channels=2,
            out_channels=3,
            kernel_size=[[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
            strides=[[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]).eval().to(device),
        int_steps=9).eval().to(device)
    model.load_state_dict(torch.load(args.load), strict=True)
    # Load models
    model = HadjHamouSVF(t0=t0, t1=t1, model=model, device=device)
    print(model)

    model.eval().to(device)

    with torch.no_grad():
        velocity = model.forward(subjects_dataset_transform)
        source_t0 = transforms_without_norm(source_subject)
        source_image = source_t0["image"][tio.DATA].unsqueeze(0).to(device)
        source_label = source_t0["label"][tio.DATA].unsqueeze(0).to(device)
        for i in range(len(subjects_dataset)):
            subject = subjects_dataset[i]
            timed_velocity = subject['age'] * velocity
            forward_flow = model.reg_model.velocity2displacement(timed_velocity)
            warped_source_image = warp(source_image.float(), forward_flow)
            warped_source_label = torch.argmax(warp(source_label.float(), forward_flow), dim=1).unsqueeze(0)
            warped_subject = tio.Subject(
                image=tio.ScalarImage(tensor=warped_source_image.detach().cpu().squeeze(0),
                                      affine=source_t0['image'].affine),
                label=tio.LabelMap(tensor=warped_source_label.squeeze(0).int().detach().cpu(),
                                   affine=source_t0['label'].affine),
                flow=tio.ScalarImage(tensor=forward_flow.squeeze(0).detach().cpu(),
                                     affine=source_t0['image'].affine)
            )
            warped_subject = reverse_transform(warped_subject)
            warped_subject['image'].save(os.path.join(result_save_path_images,  f"warped-t{i}.nii.gz"))
            warped_subject['label'].save(os.path.join(result_save_path_images,  f"df-t{i}.nii.gz"))
            warped_subject['flow'].save(os.path.join(result_save_path_images,  f"warped-t{i}_label.nii.gz"))




if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--dataset_yaml', type=str, help='Subject name', default="/home/florian/Documents/Programs/longitudinal-svf-pair/src/configs/data/atlasimages.yaml")
    argparse.add_argument('--savePath', type=str, default=".", help='Model save path')
    argparse.add_argument('--load', type=str, default=".", help='Model save path')
    args = argparse.parse_args()
    main(args)