#!/bin/bash

CHECK_POINT_OPT=
if [ "$1" == "--continue" ]; then
    CHECK_POINT_OPT="--checkpoint ./sharingan_checkpoints"
fi
echo CHECK_POINT_OPT=$INPUT_DIR_OPT

OUTPUT_DIR="../data/output/pretrained_model"
rm -rf $OUTPUT_DIR
mkdir $OUTPUT_DIR

python sharingan_train.py \
  --output_dir $OUTPUT_DIR \
  --max_epochs 150 \
  --lr 0.00004 \
  --beta1 0.5 \
  --conv_std 0.0005 \
  --ngf 32 \
  --ndf 32 \
  --batch_size 32 \
  --enable_quantization \
  --input_dir "../data/input/training" 2>&1 | tee $OUTPUT_DIR/output.log
