import monai
import torchio as tio
import os
import torch

import os
# Script to test ANTS registration to capture brain folding
# Input intensity images : t2_t<week>.00_128.nii.gz
# Input segmented images : t2-t<week>.00_128-bounti_label_{label}.nii.gz

# Current best result obtained with :
# segmentation only (all labels)
# transform = SyN
# gradientStep = 1
from src.utils.grid_utils import warp
# 21 weeks is considered as a fixed starting point
starting_week = 21
last_week = 37
# List of weeks to process
weeks = range(22,last_week+1,1)
path_flows = '/home/florian/Desktop/Result/Gholipour_2/ANTS/Flow'

output_path = './'
os.makedirs(output_path, exist_ok=True)

source = tio.ScalarImage(f'/home/florian/Documents/Dataset/Gholipour_2/affine/structural/STA{starting_week}.nii.gz')
source_label = tio.LabelMap(f'/home/florian/Documents/Dataset/Gholipour_2/affine/parcellations/STA{starting_week}-mask-brain_bounti-19.nii.gz')

one_source_label = tio.OneHot()(source_label)
spacing = torch.tensor([0.5, 0.5, 0.5])  # Example spacing, adjust as needed
for week in weeks:
    flow = tio.ScalarImage(os.path.join(path_flows, f'ants_SyN_gs1_seg_all_CC_intensity_21_on_{week}_Warp.nii.gz'))
    #flow[tio.DATA] = flow[tio.DATA][[2, 1, 0],...]
    # ANTS reverse the Y axis
    #flow[tio.DATA][1,...] = -flow[tio.DATA][1,...]
    # Rotate the image 180 degrees around Z axis
    flow = tio.CropOrPad(target_shape=source[tio.DATA].shape[1:])(flow)
    warp_intensity = warp(source[tio.DATA].unsqueeze(0).double(), (flow[tio.DATA].double()).unsqueeze(0) / spacing.view(1,3, 1, 1, 1))

    tio.ScalarImage(tensor=warp_intensity.squeeze(0), affine=source.affine).save(f'intensity_warped_t{week}.nii.gz')
    warp_label = torch.argmax(warp(one_source_label[tio.DATA].unsqueeze(0).double(), (flow[tio.DATA].double().unsqueeze(0) / spacing.view(1,3, 1, 1, 1))), dim=1)
    tio.LabelMap(tensor=warp_label, affine=source_label.affine).save(f'labels_warped_t{week}.nii.gz')
    tio.ScalarImage(tensor= (flow[tio.DATA].double()), affine=source_label.affine).save(f'flow_warp_t{week}.nii.gz')