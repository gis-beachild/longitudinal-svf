#!/bin/bash

DATASET="gholipour"
SAVE_PATH="./results"
SVF_PAIR=false
SVF_LIN=false
SVF_MLP=false


while getopts "vaunslmd:" opt; do
  case $opt in
    v) SVF_PAIR=true ;;
    l) SVF_LIN=true ;;
    m) SVF_MLP=true ;;
    d) DATASET="$OPTARG" ;;
    *) echo "Usage: $0 [-v] [-a] [-u] [-n] [-s] [-l] [-m] [-d dataset]" ;;
  esac
done

echo "File: $DATASET"

CURRENT_DIR="$(pwd)"

case "$DATASET" in
    "macaque"|"dhcp"|"ferret"|"gholipour")
        CONFIG_DATASET="$CURRENT_DIR/configs/data/${DATASET}.yaml"
        ;;
    *)
        echo "Error: unknown dataset '$DATASET'" >&2
        exit 1
        ;;
esac



if [ "$SVF_PAIR" = true ]; then
    python ./src/train.py mode=pairwise data=$DATASET
    python ./src/predict.py --mode pairwise --name "$DATASET/pairwise" --savePath $SAVE_PATH --dataset_yaml $CONFIG_DATASET --load "./results/$DATASET/pairwise/model.pth"
fi

if [ "$SVF_LIN" = true ]; then
    python ./src/train.py mode=longitudinal_linear data=$DATASET
    python ./src/predict.py --mode longitudinal_linear --savePath $SAVE_PATH --dataset_yaml $CONFIG_DATASET --load "./results/$DATASET/longitudinal_linear/model.pth"
fi

if [ "$SVF_MLP" = true ]; then
    python ./src/train.py mode=longitudinal_mlp data=$DATASET
    python ./src/predict.py --mode longitudinal_mlp --savePath $SAVE_PATH --dataset_yaml $CONFIG_DATASET --load "./results/$DATASET/longitudinal_mlp/model.pth" --load_temporal "./results/$DATASET/longitudinal_mlp/temporal_model.pth"
fi
