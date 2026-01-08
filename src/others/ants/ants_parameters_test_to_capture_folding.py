import argparse
import torch
import yaml
import pandas as pd
import torchio as tio
import os


def create_binary_masks(path_label, path):
    label_map = tio.ScalarImage(path_label)
    if label_map[tio.DATA].shape[0] == 1:
        label_map = tio.LabelMap(tensor=label_map[tio.DATA].long(), affine=label_map.affine)
        label_map = tio.OneHot()(label_map)
        label_map = tio.ScalarImage(tensor=label_map[tio.DATA].long(), affine=label_map.affine)
    # One-hot encode: shape becomes (C, D, H, W)
    filename = os.path.basename(path_label).split(".")[0]
    path_out = os.path.join(path, filename)
    os.makedirs(path_out, exist_ok=True)
    # Save each binary mask
    print(label_map.shape[0])
    for class_idx in range(label_map.shape[0]):
        class_tensor = label_map[tio.DATA][class_idx].unsqueeze(0)  # shape (1, D, H, W)
        label = tio.ScalarImage(tensor=class_tensor, affine=label_map.affine)
        save_name = os.path.join(path_out, f"{class_idx}.nii.gz")
        label.save(save_name)
    return path_out

def main(path_source, path_target, path_source_binary_mask, path_target_binary_mask, label_range, gradientStep, transform, output_prefix):
    # Construct the command
    prefix = f'{transform}_gs{gradientStep}'
    prefix += '_seg_'
    cmd = f'antsRegistration -d 3 -o {prefix} '
    intensity = False
    seg = True
    loss = 'CC'
    cmd += f'--transform {transform}[' + gradientStep + ',2,0] '
    if intensity :
        if loss == 'CC':
            cmd += f'--metric CC[{path_target},{path_source},1,3] '
        if loss == 'MI':
            cmd += f'--metric MI[{path_target},{path_source},1,32] '
    if seg:
        weighting_seg = 1.0 / len(label_range)
        for label in label_range:
            starting_lb_path = os.path.join(path_source_binary_mask, f'{label}.nii.gz')
            current_lb_path = os.path.join(path_target_binary_mask, f'{label}.nii.gz')
            cmd += f'--metric MeanSquares[{current_lb_path}, {starting_lb_path},{weighting_seg}] '

    cmd += '--convergence [200x200x100x100,1e-7,10] '
    cmd += '--shrink-factors 8x4x2x1 '
    cmd += '--smoothing-sigmas 3x2x1x0vox '
    cmd += '--verbose 1'

    print(cmd)
    os.system(cmd)
    # Apply the transformation
    cmd = f'antsApplyTransforms -d 3 -r {path_target} -i {path_source} -t {prefix}0Warp.nii.gz -o {output_prefix}_{prefix}_Warped.nii.gz '
    print(cmd)
    os.system(cmd)
    cmd = f"mv {prefix}0Warp.nii.gz {output_prefix}_{prefix}_Warp.nii.gz"
    os.system(cmd)
    temp_folder = './temp/'
    out_tensor = None
    os.makedirs(temp_folder, exist_ok=True)
    label_idx = 0
    for label in label_range:
        starting_lb_path = os.path.join(path_source_binary_mask, f'{label}.nii.gz')
        current_lb_path = os.path.join(path_target_binary_mask, f'{label}.nii.gz')
        print(starting_lb_path, current_lb_path, )
        cmd = f'antsApplyTransforms -d 3 -r {current_lb_path} -i {starting_lb_path} -t {output_prefix}_{prefix}_Warp.nii.gz -o {temp_folder}_{label}_Warped_seg.nii.gz'
        os.system(cmd)
        print(cmd)
        img = tio.ScalarImage(f'{temp_folder}_{label}_Warped_seg.nii.gz')
        data = img.data
        if out_tensor is None:
            _, d, h, w = data.shape
            out_tensor = torch.zeros((len(label_range), d, h, w))
        out_tensor[label_idx, :] = data[0, :]
        label_idx += 1
    tio.ScalarImage(tensor=out_tensor, affine=img.affine).save(f'{output_prefix}_{prefix}_Warped_seg.nii.gz')
    tio.LabelMap(tensor=torch.argmax(out_tensor, dim=0).unsqueeze(0).int(), affine=img.affine).save(f'{output_prefix}_{prefix}_Warped_label.nii.gz')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test different ANTs parameters to capture folding')
    parser.add_argument('--dataset_yaml', type=str, help='Path to the source image', required=True)
    parser.add_argument('--savePath', type=str, help='Path to the output directory', required=False, default="./")
    parser.add_argument('-g', '--gradient', help='Gradient step (string)', type=str, required=False, default='0.7')
    parser.add_argument('-tf', '--transform', help='Transform type (string)', type=str, required=False, default='SyN')
    args = parser.parse_args()
    with open(args.dataset_yaml, "r") as f:
        config = yaml.safe_load(f)

    rsize = config['rsize']
    csize = config['csize']
    name = config['name']
    num_classes = config['num_classes']
    date_format = config['date_format']
    t0 = config['t0']

    num_classes = config['num_classes']
    df = pd.read_csv(config['csv_path'])
    lst_images = []
    lst_labels = []
    for index, row in df.iterrows():
        lst_images.append(row['image'])
        lst_labels.append(row['label'])
    image_source = lst_images[t0]
    label_source = lst_labels[t0]
    lst_labels = lst_labels[:t0] + lst_labels[t0 + 1:]
    lst_images = lst_images[:t0] + lst_images[t0 + 1:]
    args = parser.parse_args()
    temp_folder = './temp/'
    os.makedirs(temp_folder, exist_ok=True)
    binary_mask_source = create_binary_masks(label_source, temp_folder)
    for i in range(len(lst_labels)):
        prefix = os.path.join(args.savePath, args.transform, os.path.basename(image_source).split(".")[0] + "_to_" + os.path.basename(lst_images[i]).split(".")[0])
        binary_masks_target = create_binary_masks(lst_labels[i], temp_folder)
        main(path_source=image_source, path_target=lst_images[i], path_source_binary_mask=binary_mask_source, path_target_binary_mask=binary_masks_target, gradientStep=args.gradient, label_range=range(0, num_classes), transform=args.transform, output_prefix=prefix)

