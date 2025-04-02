import os
from os.path import expanduser
import getpass
import shutil
import tempfile
import argparse
import glob

def wrapper_svrtk_segmentation(input, output) -> None:
    home = expanduser("~")
    username = getpass.getuser()
    output_directory = os.path.dirname(output)
    with tempfile.TemporaryDirectory(dir=home) as temp_dir:
        # copy the reconstruction file into the temporary directory
        shutil.copy(input, temp_dir)
        bounti_input = temp_dir.replace(username,'data')
        bounti_output = output_directory.replace(username,'data')
        cmd_line = 'sudo time docker run --rm  --mount type=bind,source='+home+',target=/home/data  fetalsvrtk/svrtk:general_auto_amd '
        cmd_line+= 'bash /home/auto-proc-svrtk/scripts/auto-brain-bounti-segmentation-fetal.sh '+bounti_input+' ' +bounti_output
        print(cmd_line)
        os.system(cmd_line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Beo fetal pipeline of one subject')
    parser.add_argument('-i', '--input', help='Input folder (absolute path)', type=str, required=True)
    parser.add_argument('-o', '--output', help='Output folder (absolute path)', type=str, required=True)
    args = parser.parse_args()

    # Find automatically all images in input directory
    raw_stacks = []
    files = glob.glob(os.path.join(args.input, '*.nii.gz'))
    print(files)
    for f in files:
        raw_stacks.append(f)
    print('List of input raw stacks:')
    print(raw_stacks)

    # check is output directory exists
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    for file in raw_stacks:
        wrapper_svrtk_segmentation(file, args.output)