import argparse
import logging
import os
import torch

import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader
import pandas as pd
from config import Config
from models import FlowNet3D
from src.others.sgdir.utils import get_dataset
import yaml
import torchio as tio
from src.utils.grid_utils import warp, get_reference_grid
def format_number(n, max_n):
    """
    Format a number with leading zeros based on the maximum number.

    Args:
        n:
        max_n:

    Returns:

    """
    max_digits = len(str(max_n))
    return str(n).zfill(max_digits)
def cal_time(visit_time):
    separates = visit_time.split("-")
    year = int(separates[0])
    flag_month = separates[1].split("0")
    month = int(separates[1]) if flag_month[0] != "0" else int(flag_month[1])
    flag_day = separates[2].split("0")
    day = int(separates[2]) if flag_day[0] != "0" else int(flag_day[1])

    return year*365+month*30+day

def flow_jacdet(flow):
    vol_size = flow.shape
    grid = get_reference_grid(flow).permute(0,2,3,4,1).squeeze(0)
    flow = flow.permute(0,2,3,4,1).squeeze(0)
    J = np.gradient(flow + grid)
    dx = J[0]
    dy = J[1]
    dz = J[2]
    Jdet0 = dx[:,:,:,0] * (dy[:,:,:,1] * dz[:,:,:,2] - dy[:,:,:,2] * dz[:,:,:,1])
    Jdet1 = dx[:,:,:,1] * (dy[:,:,:,0] * dz[:,:,:,2] - dy[:,:,:,2] * dz[:,:,:,0])
    Jdet2 = dx[:,:,:,2] * (dy[:,:,:,0] * dz[:,:,:,1] - dy[:,:,:,1] * dz[:,:,:,0])
    Jdet = Jdet0 - Jdet1 + Jdet2
    return Jdet


def main(args):
    with open(args.dataset_yaml, "r") as f:
        config = yaml.safe_load(f)
    rsize = config['rsize']
    csize = config['csize']
    num_classes = config['num_classes']
    date_format = config['date_format']
    name = config['name']

    data_config = {
        'rsize': rsize,
        'csize': csize,
        'num_classes': num_classes,
        'csv_path': config['csv_path']
    }

    # configurations
    torch.cuda.empty_cache()
    config = Config(f'configs/{name}.yml')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_evals = getattr(config, 'save_evals')
    save_evals_path = getattr(config, 'save_evals_path')
    batch_size = int(getattr(config, 'batch_size'))
    down_factor = int(getattr(config, 'down_factor'))

    down_channels = getattr(config, 'down_channels') if hasattr(config, 'down_channels') else [64, 128, 256, 512]
    up_channels = getattr(config, 'up_channels') if hasattr(config, 'up_channels') else [512, 256, 128, 64]
    time_emb_dim = int(getattr(config, 'time_emb_dim')) if hasattr(config, 'time_emb_dim') else 64
    decoder_only = getattr(config, 'decoder_only') if hasattr(config, 'decoder_only') else True


    result_save_path = f"./{args.savePath}/sgdir/{name}/"
    result_save_path_images = os.path.join(result_save_path, "images")
    result_save_path_flows = os.path.join(result_save_path, "flows")
    result_save_path_seg = os.path.join(result_save_path, "parcellations")

    os.makedirs(result_save_path_images, exist_ok=True)
    os.makedirs(result_save_path_seg, exist_ok=True)
    os.makedirs(result_save_path_flows, exist_ok=True)


    # defining the logger
    logging.basicConfig(filename=getattr(config, 'eval_logdir'), format='%(asctime)s %(message)s', filemode='w')
    logging.getLogger('matplotlib').setLevel(logging.ERROR)
    logging.getLogger('PIL').setLevel(logging.ERROR)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # defining the evaluation dataset
    dataset = get_dataset(config=data_config, train=False)
    loader = DataLoader(dataset, batch_size=batch_size)

    # loading the model
    network = FlowNet3D(down_channels=down_channels,
                        up_channels=up_channels,
                        time_emb_dim=time_emb_dim,
                        decoder_only=decoder_only).to(device)
    times = []
    df = pd.read_csv(data_config['csv_path'])
    for index, row in df.iterrows():
        times.append(row['age'])
    if date_format is not None:
        times = [cal_time(t) for t in times]
    start_time = times[0]
    last_time = times[-1] - start_time
    times = [(t - start_time) / (last_time) for t in times]


    network.load_state_dict(result_save_path + './model.pth')
    with torch.no_grad():
        print('[*] Evaluation started...')
        logger.debug(f'Evaluation started on {len(dataset)} samples...')
        df = pd.read_csv(data_config['csv_path'])
        data_paths = []
        for index, row in df.iterrows():
            data_paths.append((row['image'], row['label'], row['age']))
        img = tio.ScalarImage(data_paths[0][0])
        labels = tio.ScalarImage(data_paths[0][1])
        crop = tio.CropOrPad((csize[0], csize[1], csize[2]))
        reverse_crop = tio.CropOrPad((img.shape[1], img.shape[2], img.shape[3]))
        if labels[tio.DATA].shape[0] == 1:
            labels = tio.LabelMap(tensor=labels[tio.DATA].int(), affine=labels.affine)
        img = crop(img)
        labels = crop(labels)
        if labels[tio.DATA].shape[0] == 1:
            labels = tio.OneHot(num_classes)(labels)
        for batch_id, sample in enumerate(tqdm(loader)):
            I_, J_, xyz_, seg_I_, seg_J_ = sample
            I_ = I_.to(device)
            J_ = J_.to(device)
            xyz_ = xyz_.to(device)

            j=0
            for i in times:
                print(i)
                t = torch.Tensor([i]).float().to(device)
                flow = network.get_u(I_, J_, xyz_, t).cpu()
                flow = flow[:, [2, 1, 0], ...]
                Jw_seg = warp(labels[tio.DATA].float().cpu().unsqueeze(0).detach(), flow.cpu().detach())
                Jw = warp(img[tio.DATA].float().unsqueeze(0).cpu().detach(), flow.cpu().detach())

                warp_img = tio.ScalarImage(tensor=Jw.squeeze(0), affine=img.affine)
                flow_img = tio.ScalarImage(tensor=flow.squeeze(0), affine=img.affine)

                warp_seg = tio.LabelMap(tensor=torch.argmax(Jw_seg, dim=1).int(), affine=img.affine)
                warp_img = reverse_crop(warp_img)
                index_str = format_number(j, len(times))
                nb_jac_neg = np.sum(flow_jacdet(flow.cpu().detach()) <= 0)
                print(f"Number of negative Jacobian determinants: {nb_jac_neg}")
                warp_img = reverse_crop(warp_img)
                warp_seg = reverse_crop(warp_seg)
                flow_img = reverse_crop(flow_img)
                flow_img[tio.DATA] = flow_img[tio.DATA] * torch.tensor(warp_img.spacing).view(3, 1, 1, 1)
                warp_img.save(f"{result_save_path_images}/warped-t{index_str}.nii.gz")
                flow_img.save(f"{result_save_path_flows}/df-t{index_str}.nii.gz")
                warp_seg.save(f"{result_save_path_seg}/warped-t{index_str}_label.nii.gz")
                print(result_save_path_seg)
                j += 1


if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--dataset_yaml', type=str, help='Subject name', required=True)
    argparse.add_argument('--savePath', type=str, default="./", help='Save path')
    argparse.add_argument('--lambdaGrad', type=float, default=0.005, help='Lambda for gradient loss')
    argparse.add_argument('--epochs', type=int, default=3000, help='Number of epochs')
    argparse.add_argument('--train', type=bool, default=False, help='Training before inference')
    args = argparse.parse_args()
    main(args)