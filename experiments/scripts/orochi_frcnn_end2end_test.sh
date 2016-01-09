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
DATASET_NAME=$3
NET_FINAL=$4

ITERS=70000

array=( $@ )
num_basic_args=4
len=${#array[@]}
EXTRA_ARGS=${array[@]:$num_basic_args:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

LOG="experiments/logs/orochi_frcnn_${NET}_${DATASET_NAME}_test.`date +'%Y-%m-%d_%H-%M-%S'`.txt"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/orochi_test_net.py --gpu ${GPU_ID} \
  --def-root models/${NET}/orochi_frcnn_end2end \
  --net ${NET_FINAL} \
  --ds-name ${DATASET_NAME} \
  --cfg experiments/cfgs/orochi_frcnn_end2end.yml \
  ${EXTRA_ARGS}
