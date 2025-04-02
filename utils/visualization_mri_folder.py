import torch
import torchvision
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import torchio as tio
import os
import torchvision.transforms.functional as TF
import glob
# Fix for Intel Mkl (MacOS)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize dHCP dataset')
    parser.add_argument('-i', '--input', help='Input folder', type=str, required=True)
    args = parser.parse_args()

    nb_rows = 1
    nb_cols = 15
    n_size = 512

    lst_imgs = glob.glob(args.input + "/*")
    print(lst_imgs)
    lst_imgs.sort()
    nb_img_batch = nb_cols * nb_rows

    n_batch = len(lst_imgs) // nb_img_batch
    if len(lst_imgs) % nb_img_batch != 0:
        n_batch += 1

    # Torch io transforms for rescaling and resizing.
    rescale = tio.RescaleIntensity(out_min_max=(0, 1))
    transforms = tio.Compose([rescale])

    for b in range(n_batch):
        grids = []
        for i in range(nb_img_batch):
            n = b * nb_img_batch + i
            if n >= len(lst_imgs):
                slice = torch.zeros(1, n_size, n_size)
            else:
                print(str(i) + " : " + lst_imgs[b * nb_img_batch + i].split("/")[-1].split('.')[0])
                img = tio.ScalarImage(lst_imgs[n])[tio.DATA]
                img = transforms(img)
                slice = TF.rotate(img[..., int(img.shape[-1] / 2)], 90, expand=True)
            grids.append(slice)

        grids = torchvision.utils.make_grid(grids, nrow=nb_cols)
        grids = grids.permute(1, 2, 0).numpy()
        plt.imshow(grids)
        plt.savefig("./img.png")