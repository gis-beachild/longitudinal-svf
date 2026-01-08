import os
import pandas as pd
import torchio as tio
import argparse
import yaml
from gradicon import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='UniGradIcon Registration 3D Image Pair with pretrained network')
    parser.add_argument("--dataset_yaml", type=str, help='Subject name', required=True)
    parser.add_argument("--savePath", type=str, help='Output path', required=True)
    parser.add_argument('-m', '--mode', help='Unigradicon or Multigradicon', type=str, required=False, default='unigradicon')
    args = parser.parse_args()
    with open(args.dataset_yaml, "r") as f:
        config = yaml.safe_load(f)
    rsize = config['rsize']
    csize = config['csize']
    name = config['name']
    num_classes = config['num_classes']
    date_format = config['date_format']
    t0 = config['t0']
    print(num_classes)
    df = pd.read_csv(config["csv_path"])
    lst_data = []
    for index, row in df.iterrows():
        lst_data.append((row['age'], row['image'], row['label']))

    source_data = lst_data[int(t0)]
    savePath = f"./{args.savePath}/{args.mode}/{name}"
    savePathImg = os.path.join(savePath, "images")
    savePathSeg = os.path.join(savePath, "parcellations")
    savePathFlow = os.path.join(savePath, "flows")
    os.makedirs(savePathImg, exist_ok=True)
    os.makedirs(savePathSeg, exist_ok=True)
    os.makedirs(savePathFlow, exist_ok=True)

    source_subject = tio.Subject(
        image=tio.ScalarImage(source_data[1]),
        label=tio.ScalarImage(source_data[2]),
    )
    for age, img, label in lst_data:
        print(f"Processing source: {source_data[1]}, target image: {img}, target label: {label}")
        target_subject = tio.Subject(
            image=tio.ScalarImage(img),
            label=tio.ScalarImage(label),
        )

        img_name = os.path.basename(img).split(".")[0]
        source_name = os.path.basename(source_data[1]).split(".")[0]
        warped_source, warped_label_source, phi_AB = main(source_subject, target_subject, mode=args.mode)
        print("File save : ",f'{savePathImg}/{args.mode}_{source_name}_to_{img_name}.nii.gz\n')
        warped_source.save(f'{savePathImg}/{args.mode}_{source_name}_to_{img_name}.nii.gz')
        warped_label_source.save(f'{savePathSeg}/{args.mode}_{source_name}_to_{img_name}_label.nii.gz')
        phi_AB.save(f'{savePathFlow}/{args.mode}_{source_name}_to_{img_name}_flow.nii.gz')

