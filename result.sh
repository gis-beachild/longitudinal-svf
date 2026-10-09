#!/bin/bash

DATASET="gholipour"
SAVE_PATH="./results"
ANTS=false
UNIGRAD=false
SGDIR=false
NODER=false
HH=false
SVF_PAIR=false
SVF_LIN=false
SVF_MLP=false


while getopts "gausnhvlmd:" opt; do
  case $opt in
    g) GT=true ;;
    v) SVF_PAIR=true ;;
    l) SVF_LIN=true ;;
    m) SVF_MLP=true ;;
    d) DATASET="$OPTARG" ;;
    *) echo "Usage: $0 [-v] [-a] [-u] [-n] [-s] [-l] [-m] [-d dataset]" ;;
  esac
done

CURRENT_DIR="$(pwd)"

case "$DATASET" in
    "macaque"|"dhcp"|"ferret"|"gholipour")
        CONFIG_HYDRA="$CURRENT_DIR/configs/data/${DATASET}.yaml"
        ;;
    *)
        echo "Error: unknown dataset '$DATASET'" >&2
        exit 1
        ;;
esac
NORMALIZED_PATH="./results/$DATASET/gt/gi.csv"
echo $CONFIG_HYDRA
if [ "$GT" = true ]; then
    python ./src/result_script.py --dataset_yaml $CONFIG_HYDRA  --pred $SAVE_PATH/$DATASET/gt/ --rotate 90
fi

if [ "$SVF_PAIR" = true ]; then
    python ./src/result_script.py  --dataset_yaml $CONFIG_HYDRA --pred $SAVE_PATH/$DATASET/pairwise/ --rotate 90  --gi_normalized $NORMALIZED_PATH
fi

if [ "$SVF_LIN" = true ]; then
    python ./src/result_script.py --dataset_yaml $CONFIG_HYDRA --pred $SAVE_PATH/$DATASET/longitudinal_linear/ --rotate 90  --gi_normalized $NORMALIZED_PATH
fi

if [ "$SVF_MLP" = true ]; then
    python ./src/result_script.py  --dataset_yaml $CONFIG_HYDRA --pred $SAVE_PATH/$DATASET/longitudinal_mlp/ --rotate 90  --gi_normalized $NORMALIZED_PATH
fi