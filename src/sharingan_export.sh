#!/bin/bash

rm -rf ../data/output/frozen_model
python sharingan_export.py \
  --output_dir "../data/output/frozen_model" \
  --checkpoint "../data/output/pretrained_model/model.ckpt-7001"
