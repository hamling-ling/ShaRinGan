#!/bin/bash

CHECKPOINT_DIR=../data/output/pretrained_model
CHEKPOINT_FILE=$CHECKPOINT_DIR/model.ckpt-162000

python sharingan_test.py \
  --output_dir "../data/test/pretrained_model" \
  --input_dir "../data/input/evaluation" \
  --max_steps 100000 \
  --checkpoint "$CHEKPOINT_FILE"
  
