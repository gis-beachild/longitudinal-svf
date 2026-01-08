import json
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import yaml
import argparse

from nifti_seg_to_vtk import convert_nifti_labels_union_to_vtk
from monai.metrics import DiceMetric
import csv
from pathlib import Path
from matplotlib.colors import Normalize
from registration_svf.utils.grid_utils import get_reference_grid
import meshio
from utils.gyrification_index import rescale_initial_smooth_mesh_to_folded_mesh, compute_gyrification_index
import torch
import torchio as tio
import glob
from PIL import Image
from registration_svf.utils.grid_utils import displacement2grid, plt_grid
import shutil
import pystrum.pynd.ndutils as nd
from matplotlib.colors import ListedColormap
import seaborn as sns

def read_parcellations(path: Path, num_classes: int) -> torch.Tensor:
    """
    Load a label map using TorchIO and convert it to one-hot encoded tensor
    (C, X, Y, Z). Fixes bad loading by forcing integer label casting.
    """
    to_onehot = tio.OneHot(num_classes=num_classes)
    label = tio.ScalarImage(str(path))
    affine = label.affine
    if label.data.shape[0] != 1:
        label = torch.argmax(label.data, dim=0, keepdim=True)
        label = tio.LabelMap(tensor=label, affine=affine)
    else:
        # --- Always load label maps with LabelMap ---
        label = tio.LabelMap(str(path))
    print(label.data.shape)
    # --- One-hot encoding ---
    return to_onehot(label)[tio.DATA]


def Get_Ja(displacement):
    '''
    Calculate the Jacobian value at each point of the displacement map having
    size of b*h*w*d*3 and in the cubic volumn of [-1, 1]^3
    '''
    displacement = displacement.squeeze().permute(1, 2, 3, 0)
    # check inputs
    volshape = displacement.shape[:-1]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'
    # compute grid
    grid_lst = nd.volsize2ndgrid(volshape)
    grid = np.stack(grid_lst, len(volshape))
    # compute gradients
    J = torch.gradient(displacement + torch.Tensor(grid).to(displacement.device))

    dx = J[0]
    dy = J[1]
    dz = J[2]

    # compute jacobian components
    Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
    Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
    Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

    return Jdet0 - Jdet1 + Jdet2

def flow_jacdet(flow):
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

def magnitude(x):
    """
    Compute the magnitude of a tensor.
    Args:
        x: Input tensor.
    Returns:
        Magnitude of the input tensor.
    """
    mag =  torch.linalg.norm(x, dim=0)  # (D, H, W)
    mean_mag = mag.mean()  # scalar
    return mean_mag



def build_overlay_rgba(error_slice_onehot: np.ndarray,
                       alpha: float = 1) -> Image.Image:
    """
    error_slice_onehot: (C, H, W) numpy [0/1], where channel c marks error for class c.
    class_colors: list of RGB for classes 1..C-1 (background channel 0 optional).
    Returns an RGBA overlay image with transparency outside errors.
    """
    #cmap = plt.colormaps.get_cmap('tab20')
    colors = (255, 0, 0)
    C, H, W = error_slice_onehot.shape
    a = int(np.clip(alpha, 0.0, 1.0) * 255)

    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))

    for c in range(C):
        mask = error_slice_onehot[c].astype(bool)
        if not np.any(mask):
            continue
        r, g, b = colors
        layer = np.zeros((H, W, 4), dtype=np.uint8)
        layer[mask] = (r, g, b, a)
        overlay = Image.alpha_composite(overlay, Image.fromarray(layer))

    return overlay

def to_uint8(img):
    """Normalize to 0..255 uint8 for display."""
    img = img.astype(np.float32)
    mn, mx = np.min(img), np.max(img)
    if mx > mn:
        img = (img - mn) / (mx - mn)
    else:
        img = np.zeros_like(img, dtype=np.float32)
    return (img * 255).astype(np.uint8)

def get_slice_axial(img, slice):
    slice_img_ax = img.data[:, :, slice, :].squeeze().numpy()
    return slice_img_ax

def get_slice_coronal(img, slice):
    slice_img_cor = img.data[:, slice, :, :].squeeze().numpy()
    return slice_img_cor

def get_slice_sagittal(img, slice):
    slice_img_sag = img.data[:, :, :, slice].squeeze().numpy()
    return slice_img_sag

def get_slice(img, slice, plane=1):
    if plane == 0:
        slice = get_slice_sagittal(img, slice)
    elif plane == 1:
        slice = get_slice_coronal(img, slice)
    elif plane == 2:
        slice = get_slice_axial(img, slice)
    else:
        raise ValueError("Invalid plane. Choose from 'axial', 'coronal', 'sagittal'.")
    return slice

def crop_from_segmentation_mask_with_ratio(img: np.ndarray, mask: np.ndarray, ratio: float, margin: int = 4, value=0):
    """
    Crops the minimal bounding box around the mask and pads (if needed) to
    reach the desired aspect ratio, keeping the mask centered.

    Args:
        img:  (H, W) or (C, H, W) NumPy array
        mask: (H, W) binary or labeled segmentation mask
        ratio: desired aspect ratio = height / width

    Returns:
        padded_img, padded_mask
    """
    assert mask.ndim == 2, "Mask must be (H, W)"
    assert img.ndim in (2, 3), "Image must be (H, W) or (C, H, W)"
    assert ratio > 0, "Ratio must be positive"
    assert margin > 0, "Margin must be positive"
    H, W = mask.shape
    ys, xs = np.where(mask > 0)

    # If mask is empty, return original
    if len(xs) == 0 or len(ys) == 0:
        return img.copy(), mask.copy()

    # --- Minimal bounding box ---
    
    y_min, y_max = max(0, ys.min() - margin), min(ys.max() + margin, H-1) 
    x_min, x_max = max(0, xs.min() - margin), min(xs.max() + margin, W-1)
    box_h = y_max - y_min + 1
    box_w = x_max - x_min + 1

    # --- Center of mask ---
    cy = (y_min + y_max) / 2
    cx = (x_min + x_max) / 2

    # --- Adjust box to desired aspect ratio (pad in W or H) ---
    target_h = box_h
    target_w = box_w

    current_ratio = box_h / box_w

    if current_ratio > ratio:
        # Image is too tall → pad width
        target_w = int(np.ceil(box_h / ratio))
    elif current_ratio < ratio:
        # Image is too wide → pad height
        target_h = int(np.ceil(box_w * ratio))

    # --- Compute new box coordinates ---
    y1 = int(round(cy - target_h / 2))
    y2 = y1 + target_h
    x1 = int(round(cx - target_w / 2))
    x2 = x1 + target_w

    # --- Clip to image bounds ---
    y1_clip, y2_clip = max(0, y1), min(H, y2)
    x1_clip, x2_clip = max(0, x1), min(W, x2)

    # --- Crop ---
    if img.ndim == 3:
        cropped_img = img[:, y1_clip:y2_clip, x1_clip:x2_clip]
    else:
        cropped_img = img[y1_clip:y2_clip, x1_clip:x2_clip]
    cropped_mask = mask[y1_clip:y2_clip, x1_clip:x2_clip]

    # --- Pad if needed (to keep target_h × target_w) ---
    pad_top = y1_clip - y1
    pad_bottom = y2 - y2_clip
    pad_left = x1_clip - x1
    pad_right = x2 - x2_clip
    if img.ndim == 3:
        padded = np.full((img.shape[0], target_h, target_w), fill_value=value, dtype=img.dtype)
        padded[:, pad_top:pad_top+cropped_img.shape[1], pad_left:pad_left+cropped_img.shape[2]] = cropped_img
    else:
        padded = np.full((target_h, target_w), fill_value=value, dtype=img.dtype)
        padded[pad_top:pad_top+cropped_img.shape[0], pad_left:pad_left+cropped_img.shape[1]] = cropped_img

    return padded


def error_map_slice_with_color_map(slice_img_ax, error_map_ax, alpha=0.8, rotate=90):
    bg = Image.fromarray(slice_img_ax).convert("L").convert("RGBA")
    overlay = build_overlay_rgba(error_map_ax, alpha=alpha)
    overlay = Image.alpha_composite(bg, overlay)
    overlay = overlay.rotate(rotate, expand=True)
    return overlay

def segmentation_map_slice_with_color_map(img, rotate=90):
    cmap = plt.colormaps.get_cmap('tab20')
    cmap_r = cmap.reversed()
    colors = cmap_r.colors
    colors = [(0,0,0,0)] + list(colors)
    cmap_r = ListedColormap(colors, name='new_tab21')
    img = cmap_r(img % 21)[:, :, :3]  # keep RGB, drop alpha
    img = (img * 255).astype(np.uint8)
    img = Image.fromarray(img).convert("RGB")
    img = img.rotate(rotate, expand=True)
    return img

def img_map_normalized_to_uint8(img, rotate=90):
    img = to_uint8(img)
    img = Image.fromarray(img).convert("L")
    img = img.rotate(rotate, expand=True)
    return img


def crop_volume_with_mask(img, mask, margin=5):
    """
    img: torch tensor [1, H, W, D]
    mask: torch tensor [C,H,W,D] or [1,H,W,D]
    margin: margins (in voxels) to pad around bounding box
    """
    # Ensure mask is single-channel
    if mask.shape[0] > 1:
        mask = torch.argmax(mask, dim=0, keepdim=True)  # -> [1,H,W,D]

    mask_bin = mask[0] > 0  # -> [H,W,D]

    # Get bounding box from mask
    coords = mask_bin.nonzero(as_tuple=False)  # [N,3]

    x_min = int(coords[:, 0].min())
    x_max = int(coords[:, 0].max())
    y_min = int(coords[:, 1].min())
    y_max = int(coords[:, 1].max())
    z_min = int(coords[:, 2].min())
    z_max = int(coords[:, 2].max())

    # Add margins (clamped to volume size)
    H, W, D = mask.shape[1:]

    x_min = max(0, x_min - margin)
    y_min = max(0, y_min - margin)
    z_min = max(0, z_min - margin)

    x_max = min(H, x_max + margin)
    y_max = min(W, y_max + margin)
    z_max = min(D, z_max + margin)

    # Crop using bounding box
    cropped_img = img[:, x_min:x_max, y_min:y_max, z_min:z_max]
    cropped_mask = mask[:, x_min:x_max, y_min:y_max, z_min:z_max]

    return cropped_img, cropped_mask, (x_min, x_max, y_min, y_max, z_min, z_max)

def main(args):
    ratio = 0.9
    rotate = args.rotate
    with open(args.dataset_yaml, "r") as f:
        config = yaml.safe_load(f)
    name_dataset = config["name"]
    rsize = config['rsize']
    csize = config['csize']
    t0 = config['t0']
    t1 = config['t1']
    plane = args.plane
    num_classes = config['num_classes']
    csv_path = config['csv_path']
    labels_config = config['label_metadata']
    labels_config = json.load(open(labels_config))
    cortex_labels = labels_config['CortexIndex']
    labelsNames = labels_config['class_labels']
    name_methods = args.pred.split("/")[-2]
    # Save paths for results
    parentPath = os.path.join("../methods/results", name_dataset, name_methods)
    os.makedirs(parentPath, exist_ok=True)
    preds_parcellations = glob.glob(os.path.join(args.pred, "parcellations", "*.nii*"))
    preds_images = glob.glob(os.path.join(args.pred, "images","*.nii*"))
    preds_flows = glob.glob(os.path.join(args.pred, "flows", "*.nii*"))
    preds_images.sort()
    preds_flows.sort()
    preds_parcellations.sort()
    print("Number of imgs :", len(preds_images), "--", len(preds_parcellations), "--", len(preds_flows))
    print(csv_path)
    df = pd.read_csv(csv_path)

    lst_data_gt = []
    for index, row in df.iterrows():
        lst_data_gt.append((row['age'], row['image'], row['label']))
    # Remove the t0 element if not included in the predictions
    initial_paths = lst_data_gt[0]

    if len(preds_parcellations) == len(lst_data_gt) - 1:
        lst_data_gt = lst_data_gt[1:]
    else:
        lst_data_gt = lst_data_gt

    #rescale_int = tio.transforms.RescaleIntensity(percentiles=(0, 95), masking_method='label')
    #rescale_int = tio.transforms.RescaleIntensity(percentiles=(65, 99))
    rescale_int = tio.Compose([
        tio.transforms.RescaleIntensity(percentiles=(0.3, 98), masking_method='label'),
        tio.Clamp(out_min=0, out_max=1)])
    #rescale_int = tio.transforms.RescaleIntensity(percentiles=(0, 93), masking_method='label')
    # Step 1.5: Create slices of images if needed
    if not args.img_slice:
        save_path = os.path.join(parentPath, "image_slices")
        os.makedirs(save_path, exist_ok=True)
        for i in range(len(preds_images)):
            output_path = os.path.join(save_path, f"slice_{i}.png")
            img = tio.ScalarImage(preds_images[i])
            mask = tio.ScalarImage(preds_parcellations[i])[tio.DATA]
            if mask.shape[0] != 1:
                mask = torch.argmax(mask, dim=0).unsqueeze(0)
            subject = tio.Subject(
                image=img,
                label = tio.LabelMap(tensor=mask, affine=img.affine),
            )
            subject= rescale_int(subject)
            slice_idx = subject['image'][tio.DATA].shape[-1 - plane] // 2
            img = get_slice(subject['image'][tio.DATA].float(), slice_idx, plane=plane)
            mask = np.max(mask.squeeze().numpy(), axis=(2 - plane))
            img = crop_from_segmentation_mask_with_ratio(img, mask, ratio=ratio)
            img = img_map_normalized_to_uint8(img, rotate=rotate)
            img.save(output_path)

    # Step 2 : Compute the dice score for all the parcellations
    if not args.dice:
        header = ["time", "mDice", "cortex"] + labelsNames
        dice = DiceMetric(include_background=True, reduction="none", ignore_empty=False)
        processed_rows = 0
        path_csv = os.path.join(parentPath, f"dice.csv")
        save_path_error = os.path.join(parentPath, "error_seg_slices")
        save_path_seg = os.path.join(parentPath, "seg_slices")
        os.makedirs(save_path_error, exist_ok=True)
        os.makedirs(save_path_seg, exist_ok=True)
        with open(path_csv, mode="w", newline="") as f:
            writer = csv.writer(f, delimiter=" ")
            writer.writerow(header)
            for i in range(len(preds_parcellations)):
                print(preds_parcellations[i])
                pred_1h = read_parcellations(preds_parcellations[i], num_classes=num_classes)
                print(lst_data_gt[i][2])
                gt_1h = read_parcellations(lst_data_gt[i][2], num_classes=num_classes)
                rst = dice(y_pred=pred_1h.unsqueeze(0), y=gt_1h.unsqueeze(0)).cpu().numpy()
                dice.reset()
                # rst shape: (B=1, C, ...)
                per_label = rst[0]  # (C,)
                mdice = float(np.mean(per_label))
                cortex_values = []
                for c in cortex_labels:
                    cortex_values.append(float(per_label[c]))
                cortex_dice = float(np.mean(cortex_values))
                row = [lst_data_gt[i][0], mdice, cortex_dice] + [float(per_label[c]) for c in range(per_label.shape[0])]
                writer.writerow(row)
                processed_rows += 1
                error_map = (pred_1h != gt_1h).float()
                slice_idx = error_map.shape[-1 - plane] // 2
                error_map = get_slice(error_map, slice_idx, plane=plane)
                pred_argmax = torch.argmax(pred_1h, axis=0).unsqueeze(0)
                mask = np.max(pred_argmax.squeeze().numpy(), axis=(2-plane))
                slice_seg = get_slice(pred_argmax, slice_idx, plane=plane)

                img = tio.ScalarImage(preds_images[i])
                mask = tio.ScalarImage(preds_parcellations[i])[tio.DATA]

                if mask.shape[0] != 1:
                    mask = torch.argmax(mask, dim=0).unsqueeze(0)
                subject = tio.Subject(
                    image=img,
                    label=tio.LabelMap(tensor=mask, affine=img.affine),
                )
                subject = rescale_int(subject)
                img = get_slice(subject['image'][tio.DATA].float(), slice_idx, plane=plane)
                img = to_uint8(img)
                mask = np.max(pred_argmax.squeeze().numpy(), axis=(2 - plane))
                img = crop_from_segmentation_mask_with_ratio(img, mask, ratio=ratio)
                error_map = crop_from_segmentation_mask_with_ratio(error_map, mask, ratio=ratio)
                slice_seg = crop_from_segmentation_mask_with_ratio(slice_seg, mask, ratio=ratio)
                slice_seg = segmentation_map_slice_with_color_map(slice_seg, rotate=rotate)
                error_map_img = error_map_slice_with_color_map(img, error_map, alpha=0.8, rotate=rotate)
                output_path = os.path.join(save_path_error, f"slice_{i}.png")
                error_map_img.save(output_path)
                output_path = os.path.join(save_path_seg, f"slice_{i}.png")
                slice_seg.save(output_path)
                print(output_path)
    # Step 3 : Compute the flow metrics for all the flows
    if not args.flow:
        path_csv = os.path.join(parentPath, f"flow.csv")
        header = ["time", "magnitude", 'Neg_J']
        image_save_path = os.path.join(parentPath, "flow_slices")
        grid_save_path = os.path.join(parentPath, "grid_slices")
        os.makedirs(image_save_path, exist_ok=True)
        os.makedirs(grid_save_path, exist_ok=True)
        all_slices = []  # store slices for the global plot
        # Colormap and normalization
        norm = Normalize(vmin=-4, vmax=4)
        with open(path_csv, mode="w", newline="") as f:
            writer = csv.writer(f, delimiter=" ")
            writer.writerow(header)
            for i in range(len(preds_flows)):
                flow = tio.ScalarImage(preds_flows[i])
                spacing = torch.Tensor(flow.spacing).view(3, 1, 1, 1)
                age = lst_data_gt[i][0]
                flow = flow[tio.DATA] / spacing  # to voxel space
                flow = flow.float()
                row = [age]
                mag = magnitude(flow * spacing).item()
                row.append(mag)
                # Compute determinant
                mask = tio.ScalarImage(preds_parcellations[i])[tio.DATA]
                if mask.shape[0] != 1:
                    mask = torch.argmax(mask, dim=0).unsqueeze(0)
                jdet = Get_Ja(flow.unsqueeze(0)).numpy()
                nb_jac_neg = np.sum(jdet < 0)
                nb_elem_volume = np.count_nonzero(mask>0)
                row.append(nb_jac_neg * 100. / nb_elem_volume)
                slice_idx = jdet.shape[-3] // 2
                jdet_slice = get_slice(torch.Tensor(jdet).unsqueeze(0), slice_idx, plane=plane)
                # Mask negative values
                neg_mask = jdet_slice < 0
                mask = np.max(mask.squeeze(0).numpy(), axis=(2 - plane))
                jdet_log = np.log(np.clip(jdet_slice, 1e-8, None))
                jdet_log = crop_from_segmentation_mask_with_ratio(jdet_log, mask=mask, ratio=ratio)
                neg_mask = crop_from_segmentation_mask_with_ratio(neg_mask, mask=mask, ratio=ratio)

                # Rotate for correct orientation
                jdet_log = np.rot90(jdet_log, 1)
                neg_mask = np.rot90(neg_mask, 1)
        
                norm = Normalize(vmin=-4, vmax=4)
                cmap = sns.diverging_palette(145, 300, as_cmap=True)
                gray_map = cmap(norm(jdet_log))[:, :, :3]  # drop alpha
                gray_img = (gray_map * 255).astype(np.uint8)
                gray_pil = Image.fromarray(gray_img)
                # --- Create red overlay from neg_mask ---
                overlay = np.zeros((*neg_mask.shape, 4), dtype=np.uint8)
                overlay[..., 0] = 255  # red channel
                overlay[..., 3] = (neg_mask * 200).astype(np.uint8)  # alpha (0–255)
                overlay_pil = Image.fromarray(overlay)
                # --- Composite overlay on grayscale ---
                base = gray_pil.convert("RGBA")
                composite = Image.alpha_composite(base, overlay_pil)
                composite.save(os.path.join(image_save_path, f"slice_{i}.png"))
                flow_slice = get_slice(flow, slice=slice_idx, plane=plane)
                masked_data = crop_from_segmentation_mask_with_ratio(flow_slice, mask=mask, ratio=ratio)
                masked_data = torch.Tensor(masked_data).unsqueeze(0)
                masked_data = masked_data[:, :-1, ...]
                
                xy = displacement2grid(masked_data).squeeze(0)
                grid_plot, _ = plt_grid(xy=xy, ratio=ratio)
                grid_plot = grid_plot.rotate(180)
                grid_plot.save(os.path.join(grid_save_path, f"slice_{i}.png"))
                
                all_slices.append(np.array(composite))
                # Write CSV row
                writer.writerow(row)
        # --- Global figure with all slices in one row ---
        n = len(all_slices)
        if n > 0:
            fig, axs = plt.subplots(1, n, figsize=(4*n, 4))
            if n == 1:
                axs = [axs]  # make iterable if only one subplot
            for i in range(n):
                im = axs[i].imshow(all_slices[i], cmap=cmap, norm=norm)
                axs[i].axis('off')
            fig.colorbar(im, ax=axs, orientation='vertical', fraction=0.02)
            plt.savefig(os.path.join(image_save_path, f'jacobian_all.png'), bbox_inches='tight')

    # Step 1: Create vtk files for cortex labels
    if args.vtk:
        vtk_path = os.path.join(args.pred, "vtk")
        os.makedirs(vtk_path, exist_ok=True)
        for p_seg in preds_parcellations:
            img = tio.ScalarImage(p_seg)
            output_path = p_seg.replace('parcellations', 'vtk').replace('nii.gz', 'vtk')
            if img.data.shape[0] != 1:
                os.makedirs("./temp/", exist_ok=True)
                img_tensor = torch.argmax(img.data, dim=0).unsqueeze(0)
                img = tio.LabelMap(tensor=img_tensor.int(), affine=img.affine)
                img.save('./temp/img.nii.gz')
                p_seg = './temp/img.nii.gz'
            convert_nifti_labels_union_to_vtk(p_seg, cortex_labels, output_path)
        if os.path.exists("./temp/"):
            shutil.rmtree("./temp/")
    # Step 4 : Compute the GI metric
    if args.gi:
        header = ["time", "GI", "Normalized_GI"]
        path_csv = os.path.join(parentPath, f"gi.csv")
        initial_smooth_mesh = meshio.read(initial_paths[2].replace('parcellations', 'vtk').replace('.nii.gz', '.vtk'))
        with open(path_csv, mode="w", newline="") as f:
            writer = csv.writer(f, delimiter=" ")
            writer.writerow(header)
            vtk_files = glob.glob(os.path.join(args.pred, "vtk", "*.vtk"))
            vtk_files.sort()
            for i in range(len(vtk_files)):
                print(vtk_files[i])
                folded_mesh = meshio.read(vtk_files[i])
                # rescale initial smooth brain mesh onto the folded brain mesh
                rescaled_initial_smooth_mesh = rescale_initial_smooth_mesh_to_folded_mesh(initial_smooth_mesh,
                                                                                          folded_mesh)
                # compute gyrification index
                GI = compute_gyrification_index(rescaled_initial_smooth_mesh, folded_mesh)
                row = [lst_data_gt[i][0]]
                row.append(GI)
                if args.gi_normalized != "":
                    data = pd.read_csv(args.gi_normalized, sep=" ")
                    gi_gt = data[data['time'] == lst_data_gt[i][0]].values[0][1]
                    row.append(GI / gi_gt)
                writer.writerow(row)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Processing results of registration experiments")
    parser.add_argument('--dataset_yaml', type=str, help='Path to dataset yaml file', default="/home/florian/Documents/Programs/longitudinal-svf-pair/src/ours-svf/svf-pair-long/configs/data/furet.yaml")
    parser.add_argument('--pred', type=str, required=False, help='Predicted segmentation file path', default="/home/florian/Desktop/Result/furet/ANTS/")
    parser.add_argument('--dice', action='store_true', help='Whether to compute dice score')
    parser.add_argument('--vtk', action='store_true', help='Whether to create vtk files')
    parser.add_argument('--flow', action='store_true', help='Whether to create flow files')
    parser.add_argument('--gi', action='store_true', help='Whether to create GI file')
    parser.add_argument('--img_slice', action='store_true', help='Whether slice of images')
    parser.add_argument('--plane', help='Whether slice of images', type=int, default=0)
    parser.add_argument('--gi_normalized', type=str, help='Path to the csv file containing the normalized GI values', default="")
    parser.add_argument('--rotate', type=int, help='Rotate slide', default=90)

    args = parser.parse_args()
    print(args)
    main(args)

