#!/bin/bash

DATASET="atlasimages"
SAVE_PATH="./results"
ANTS=false
UNIGRAD=false
SGDIR=false
NODER=false
SVF_PAIR=false
SVF_LIN=false
SVF_MLP=false


while getopts "aunslmd:" opt; do
  case $opt in
    a) ANTS=true ;;
    u) UNIGRAD=true ;;
    n) NODER=true ;;
    s) SVF_PAIR=true ;;
    l) SVF_LIN=true ;;
    m) SVF_MLP=true ;;
    d) DATASET="$OPTARG" ;;
    *) echo "Usage: $0 [-v] [-a] [-u] [-n] [-s] [-l] [-m] [-d dataset]" ;;
  esac
done

echo "File: $DATASET"

if [["$DATASET" == "atlasimages"]]; then
  CONFIG_DATASET="data/atlasimages.yaml"
elif [["$DATASET" == "dhcp"]]; then
  CONFIG_DATASET="data/dhcpatlas.yaml"
elif [["$DATASET" == "ferret"]]; then
  CONFIG_DATASET="data/ferret.yaml"
elif [["$DATASET" == "gholipour"]]; then
  CONFIG_DATASET="data/gholipour.yaml"
else
    echo "Error: unknown dataset '$DATASET'" >&2
    exit 1
fi

if [[ "$ANTS" == true ]]; then
    python ./src/others/ants/ants_parameters_test_to_capture_folding.py --savePath $SAVE_PATH --dataset_yaml $CONFIG_DATASET
fi

if [[ "$UNIGRAD" == true ]]; then
    python ./src/others/ui/unigradicon_dataset_script.py --savePath $SAVE_PATH --dataset_yaml $CONFIG_DATASET
fi

if [[ "$sgdir" == true ]]; then
    python ./src/others/sgdir/main.py --savePath $SAVE_PATH --dataset_yaml $CONFIG_DATASET
fi

if [[ "$NODER" == true ]]; then
    python ./src/others/noder/main.py --savePath $SAVE_PATH --dataset_yaml $CONFIG_DATASET
fi

if [[ "$SVF_PAIR" == true ]]; then
    python ./src/others/noder/main.py --savePath $SAVE_PATH --dataset_yaml $CONFIG_DATASET
fi

if [[ "$NODER" == true ]]; then
    python ./src/others/noder/main.py --savePath $SAVE_PATH --dataset_yaml $CONFIG_DATASET
fi

if [[ "$SVF_PAIR" == true ]]; then
    python ./src/ours-svf/svf-pair/train.py --savePath $SAVE_PATH --dataset_yaml $CONFIG_DATASET
    python ./src/ours-svf/svf-pair/predict.py --savePath $SAVE_PATH --dataset_yaml $CONFIG_DATASET
    python ./src/others/hh/main.py --savePath $SAVE_PATH --dataset_yaml $CONFIG_DATASET --load "./results/$DATASET/"svf_pair"/model.pth"
    python ./home/florian/Documents/Programs/longitudinal-svf/src/ours-svf/svf-long/predict.py --savePath $SAVE_PATH --dataset_yaml $CONFIG_DATASET --load "./results/$DATASET/"svf_pair"/model.pth" --time_mode svf_int
fi

if [[ "$SVF_LIN" == true ]]; then
    python /home/florian/Documents/Programs/longitudinal-svf/src/ours-svf/svf-long/train.py --config-name config_long_lin data=$DATASET
    python ./home/florian/Documents/Programs/longitudinal-svf/src/ours-svf/svf-long/predict.py --savePath $SAVE_PATH --dataset_yaml $CONFIG_DATASET --load "./results/$DATASET/"svf_linear"/model.pth" --time_mode svf_lin
fi

if [[ "$SVF_MLP" == true ]]; then
    python ./home/florian/Documents/Programs/longitudinal-svf/src/ours-svf/svf-long/train.py --config-name config_long_mlp data=$DATASET
    python ./home/florian/Documents/Programs/longitudinal-svf/src/ours-svf/svf-long/predict.py --savePath $SAVE_PATH --dataset_yaml CONFIG_DATASET --load "./results/$DATASET/"svf_mlp"/model.pth" --time_mode svf_mlp
fi

