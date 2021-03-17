#!/bin/bash

OUTPUT_DIR="../data/output/pretrained_model"
rm -rf $OUTPUT_DIR
mkdir $OUTPUT_DIR

python sharingan_train.py \
  --output_dir $OUTPUT_DIR \
  --max_epochs 10 \
  --lr 0.0001 \
  --beta1 0.05 \
  --conv_std 0.005 \
  --ngf 32 \
  --ndf 32 \
  --batch_size 32 \
  --input_dir "../data/input/training" 2>&1 | tee $OUTPUT_DIR/output.log
