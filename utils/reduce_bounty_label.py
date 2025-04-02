import os
import glob
import argparse
import torchio as tio


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Reduce/Change the label's index of a list of segmentation")
    parser.add_argument('--path', type=str, help='Path to the directory containing the segmentation files', required=True)
    parser.add_argument('--output-path', type=str, help='Path to the directory where the new label will be stored', required=True)
    parser.add_argument('--list', type=str, help='List of labels to change (pair index: new index)', required=True)
    args = parser.parse_args()
    labels_list = args.list.split(",")
    label_pairs = {}

    for pair in labels_list:
        index, new_index = pair.split(":")
        label_pairs[int(index)] = int(new_index)
    labels = glob.glob(args.path)

    for label in labels:
        file_name = label.split("/")[-1]
        segmentation = tio.LabelMap(label)
        new_segmentation = segmentation[tio.DATA].clone() * 0
        for key, value in label_pairs.items():
            new_segmentation[segmentation[tio.DATA] == key] = value
        segmentation[tio.DATA] = new_segmentation
        segmentation.save(os.path.join(args.output_path, file_name))