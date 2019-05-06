#!/bin/bash

python sharingan_cvt.py \
  --output_dir "../data/cvt/pretrained_model" \
  --input_dir "../data/input/evaluation" \
  --max_steps 100000 \
  --checkpoint "../data/output/pretrained_model/model.ckpt-33000"
