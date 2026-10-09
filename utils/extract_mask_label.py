import torchio as tio
import os
import glob
# Load your label map (multi-class)
path = '/home/florian/Documents/Programs/longitudinal-svf/macaque/fusionned_segmentation'
lst_file_to_convert = glob.glob(os.path.join(path, "*.nii.gz"))

for file in lst_file_to_convert:
    label_map = tio.ScalarImage(file)
    # One-hot encode: shape becomes (C, D, H, W)
    filename = os.path.basename(file).split(".")[0]
    path_binary_mask = path.replace("fusionned_segmentation", f"binary_mask/{filename}")
    os.makedirs(path_binary_mask, exist_ok=True)
    # Save each binary mask
    for class_idx in range(label_map.shape[0]):
        class_tensor = label_map[tio.DATA][class_idx].unsqueeze(0)  # shape (1, D, H, W)
        label = tio.ScalarImage(tensor=class_tensor, affine=label_map.affine)
        save_name = os.path.join(path_binary_mask, f"{class_idx}.nii.gz")
        label.save(save_name)