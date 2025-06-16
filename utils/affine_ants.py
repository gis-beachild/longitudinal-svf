import os
import ants
import argparse
import glob


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Affine transformation of a set of images to a target image')
    parser.add_argument('-s', '--source_dir', help='Source image', type=str, required=True)
    parser.add_argument('-t', '--target', help='Target image', type=str, required=True)
    parser.add_argument('-o', '--output', help='Output directory', type=str, required=True)
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    lst_source_imgs = glob.glob(args.source_dir + "/*.nii.gz")
    target = ants.image_read(args.target)
    for path_img in lst_source_imgs:
        source = ants.image_read(path_img)
        registration_result = ants.registration(target,
                                                source,
                                                type_of_transform='Affine',
                                                aff_metric='mattes'
                                                )
        warped_source = ants.apply_transforms(fixed=target, moving=source,
                                              transformlist=registration_result["fwdtransforms"])
        ants.image_write(warped_source, args.output + "/" + path_img.split("/")[-1])