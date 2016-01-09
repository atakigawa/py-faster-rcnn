#!/bin/bash
# Usage:
# ./experiments/scripts/orochi_faster_rcnn_end2end.sh GPU NET [--set ...]
# Example:
# ./experiments/scripts/orochi_faster_rcnn_end2end.sh 0 ZF \
#   --set EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400,500,600,700]"

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
NET=$2
DATASET_TRAIN=$3

ITERS=70000

array=( $@ )
num_basic_args=3
len=${#array[@]}
EXTRA_ARGS=${array[@]:$num_basic_args:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

LOG="experiments/logs/orochi_frcnn_${NET}_${DATASET_TRAIN}.`date +'%Y-%m-%d_%H-%M-%S'`.txt"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

NET_INIT=data/imagenet_models/${NET}.v2.caffemodel

time ./tools/orochi_train_net.py --gpu ${GPU_ID} \
  --solver models/${NET}/faster_rcnn_end2end/solver.prototxt \
  --weights ${NET_INIT} \
  --imdb ${DATASET_TRAIN} \
  --iters ${ITERS} \
  --cfg experiments/cfgs/orochi_frcnn_end2end.yml \
  ${EXTRA_ARGS}
