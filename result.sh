#!/bin/bash

DATASET="macaque"
SAVE_PATH="./results"
ANTS=false
UNIGRAD=false
SGDIR=false
NODER=false
HH=false
SVF_PAIR=false
SVF_INT=false
SVF_LIN=false
SVF_MLP=false


while getopts "ausnvilmd:" opt; do
  case $opt in
    a) ANTS=true ;;
    u) UNIGRAD=true ;;
    s) SGDIR=true;;
    n) NODER=true ;;
    h) HH=true;;
    v) SVF_PAIR=true ;;
    i) SVF_INT=true ;;
    l) SVF_LIN=true ;;
    m) SVF_MLP=true ;;
    d) DATASET="$OPTARG" ;;
    *) echo "Usage: $0 [-v] [-a] [-u] [-n] [-s] [-l] [-m] [-d dataset]" ;;
  esac
done


if [["$DATASET" == "macaque"]]; then
  CONFIG_HYDRA="data/macaque.yaml"
elif [["$DATASET" == "dhcp"]]; then
  CONFIG_HYDRA="data/dhcp.yaml"
elif [["$DATASET" == "ferret"]]; then
  CONFIG_HYDRA="data/ferret.yaml"
elif [["$DATASET" == "gholipour"]]; then
  CONFIG_HYDRA="data/gholipour.yaml"
else
    echo "Error: unknown dataset '$DATASET'" >&2
    exit 1
fi

NORMALIZED_PATH="./results/$DATASET/GT/gi.csv"

if [[ "GT" == true ]]; then
    python ./utils/result_script.py --savePath $SAVE_PATH --dataset_yaml $DATASET  --pred /result/$DATASET/GT/ --rotate 90
fi

if [[ "$ANTS" == true ]]; then
    python ./utils/result_script.py --savePath $SAVE_PATH --dataset_yaml $DATASET --pred ./result/$DATASET/ANTS/ --rotate 90  --gi_normalized $NORMALIZED_PATH
fi

if [[ "$UNIGRAD" == true ]]; then
    python ./utils/result_script.py --savePath $SAVE_PATH --dataset_yaml $DATASET --pred ./result/$DATASET/UI/ --rotate 90  --gi_normalized $NORMALIZED_PATH
fi

if [[ "$SGDIR" == true ]]; then
    python ./utils/result_script.py --savePath $SAVE_PATH --dataset_yaml $DATASET --pred ./result/$DATASET/SGDIR/ --rotate 90  --gi_normalized $NORMALIZED_PATH
fi

if [[ "$NODER" == true ]]; then
    python ./utils/result_script.py --savePath $SAVE_PATH --dataset_yaml $DATASET --pred ./result/$DATASET/NODER/ --rotate 90  --gi_normalized $NORMALIZED_PATH
fi

if [[ "$HH" == true ]]; then
    python ./utils/result_script.py --savePath $SAVE_PATH --dataset_yaml $DATASET --pred ./result/$DATASET/HH/ --rotate 90  --gi_normalized $NORMALIZED_PATH
fi

if [[ "$SVF_PAIR" == true ]]; then
    python ./utils/result_script.py --savePath $SAVE_PATH --dataset_yaml $DATASET --pred ./result/$DATASET/SVF_PAIR/ --rotate 90  --gi_normalized $NORMALIZED_PATH
fi

if [[ "$SVF_INT" == true ]]; then
    python ./utils/result_script.py --savePath $SAVE_PATH --dataset_yaml $DATASET --pred ./result/$DATASET/SVF_INT/ --rotate 90  --gi_normalized $NORMALIZED_PATH
fi

if [[ "$SVF_LIN" == true ]]; then
    python ./utils/result_script.py --savePath $SAVE_PATH --dataset_yaml $DATASET --pred ./result/$DATASET/SVF_LIN/ --rotate 90  --gi_normalized $NORMALIZED_PATH
fi

if [[ "$SVF_MLP" == true ]]; then
    python ./utils/result_script.py --savePath $SAVE_PATH --dataset_yaml $DATASET --pred ./result/$DATASET/SVF_MLP/ --rotate 90  --gi_normalized $NORMALIZED_PATH
fi