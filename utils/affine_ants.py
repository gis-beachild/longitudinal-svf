import os
import ants
import argparse
import glob
import torchio as tio
import numpy as np
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Affine transformation of a set of images to a target image')
    parser.add_argument('-s', '--source_dir', help='Source image dir', type=str, required=True)
    parser.add_argument('-sl', '--source_label', help='Source labels dir', type=str, required=True)
    parser.add_argument('-t', '--target', help='Target image', type=str, required=True)
    parser.add_argument('-tl', '--target_label', help='Target labels', type=str, required=True)
    parser.add_argument('-o', '--output', help='Output directory', type=str, required=True)
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    lst_source_imgs = glob.glob(args.source_dir + "/*.nii.gz")
    lst_source_labels_imgs = glob.glob(args.source_label + "/*.nii.gz")
    lst_source_imgs.sort()
    lst_source_labels_imgs.sort()
    print("List of source images:")
    print(lst_source_imgs)
    print("List of source label images:")
    print(lst_source_labels_imgs)
    target = ants.image_read(args.target)
    target_label = ants.image_read(args.target_label)
    for i in range(len(lst_source_imgs)):
        path_img = lst_source_imgs[i]
        path_source_img = lst_source_labels_imgs[i]
        source = ants.image_read(path_img)


        registration_result = ants.registration(target,
                                                source,
                                                type_of_transform='Affine',
                                                aff_metric='mattes'
                                                )
        warped_source = ants.apply_transforms(fixed=target, moving=source,
                                              transformlist=registration_result["fwdtransforms"])
        ants.image_write(warped_source, args.output + "/" + path_img.split("/")[-1])

        label_tio = tio.LabelMap(path_source_img)
        ants_label = ants.image_read(path_source_img)
        one_hot_label = tio.OneHot()(label_tio)
        source_label = one_hot_label[tio.DATA].numpy().astype(np.float32)
        transformed_labels = np.zeros_like(source_label)
        for c in range(source_label.shape[0]):
            channel_label = ants.from_numpy(source_label[c], origin=ants_label.origin, spacing=ants_label.spacing, direction=ants_label.direction)
            warped_source = ants.apply_transforms(fixed=target, moving=channel_label,
                                                  transformlist=registration_result["fwdtransforms"])
            transformed_labels[c] = warped_source.numpy()
            #print("Number of voxels in class ", c, np.sum(source_label[c]), " after warping ", np.sum(transformed_labels[c]))
        print(transformed_labels.shape)
        transformed_label = torch.from_numpy(transformed_labels)
        tio.ScalarImage(tensor=transformed_label, affine=label_tio.affine).save(args.output + "/" + path_source_img.split("/")[-1])




