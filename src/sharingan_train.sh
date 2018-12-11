#!/bin/bash

CHECK_POINT_OPT=
if [ "$1" == "--continue" ]; then
    CHECK_POINT_OPT="--checkpoint ./sharingan_checkpoints"
fi
echo CHECK_POINT_OPT=$INPUT_DIR_OPT

python sharingan_train.py \
  --output_dir "sharingan_checkpoints" \
  --max_epochs 200 \
  --input_dir "../data/input/training" $CHECK_POINT_OPT



