import os
import torch
import pandas as pd
from typing import Callable, List
import torchio as tio
import pytorch_volumetric as pv

def normalize_to_0_1(volume):
    '''
        Normalize volume to 0-1 range
    '''
    max_val = volume.max()
    min_val = volume.min()
    return (volume - min_val) / (max_val - min_val)

import torch.nn.functional as F
def compute_sdf_3d(mask):
    """
    label: (B,C,D,H,W) float binary
    Retourne: same shape, float SDF différentiable sur GPU
    """
    """
    Calcule la Signed Distance Function (SDF) 3D différentiable.
    Args:
        mask (torch.Tensor): Tenseur binaire de forme (B, C, D, H, W).
    Returns:
        torch.Tensor: SDF de forme (B, C, D, H, W).
    """
    dx = torch.abs(mask[:, :, 1:, :, :] - mask[:, :, :-1, :, :])
    dy = torch.abs(mask[:, :, :, 1:, :] - mask[:, :, :, :-1, :])
    dz = torch.abs(mask[:, :, :, :, 1:] - mask[:, :, :, :, :-1])
    edges = F.pad(dx, (0,0,0,0,0,1)) + F.pad(dy, (0,0,0,1,0,0)) + F.pad(dz, (0,1,0,0,0,0))
    sdf = 1.0 / (edges + 1e-6)
    # Pondération signée : + à l'intérieur (masque > 0.5), - à l'extérieur
    sign = torch.where(mask > 0.5, 1.0, -1.0)
    return sdf * sign

def get_weight_from_sdm(sdm, max_distance=5):
    weight = torch.exp(- (sdm / max_distance)**2)
    return weight  # Taille (1, 1, D, H, W)

def subjects_from_csv(dataset_path: str, age=True, lambda_age: Callable = lambda x: x) -> List[tio.Subject]:
    """
    Function to create a list of subjects from a csv file
    Args:
        dataset_path: path to the csv file (First column should be the path to the image, second column the path to the label and third column the age)
        age: boolean to include age in the subject
        lambda_age: function to apply to the age
    """
    subjects = []
    df = pd.read_csv(dataset_path)
    for index, row in df.iterrows():
        if age is False:
            subject = tio.Subject(
                image=tio.ScalarImage(row['image']),
                label=tio.LabelMap(row['label'])
            )
        else:
            age_value = lambda_age(row['age']) if lambda_age is not None else row['age']
            subject = tio.Subject(
                image=tio.ScalarImage(row['image']),
                label=tio.LabelMap(row['label']),
                age=age_value
            )
        subjects.append(subject)
    return subjects


def create_new_versioned_directory(base_name='', start_version=0):
    '''
        Create a new versioned directory
    '''
    # Check if version_0 exists
    version = start_version
    while os.path.exists(f'{base_name}_{version}'):
        version += 1
    new_version = f'{base_name}_{version}'
    os.makedirs(new_version)
    print(f'Created repository version: {new_version}')
    return new_version


# Create a new directory recursively if it does not exist
def create_directory(directory):
    '''
        Create a new directory recursively if it does not exist
    '''
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f'Created directory: {directory}')
    return directory



