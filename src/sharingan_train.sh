#!/bin/bash

CHECK_POINT_OPT=
if [ "$1" == "--continue" ]; then
    CHECK_POINT_OPT="--checkpoint ./sharingan_checkpoints"
fi
echo CHECK_POINT_OPT=$INPUT_DIR_OPT

python sharingan_train.py \
  --output_dir "../data/output/pretrained_model" \
  --max_epochs 600 \
  --lr 0.0001 \
  --ngf 32 \
  --ndf 32 \
  --batch_size 64 \
  --input_dir "../data/input/training"



