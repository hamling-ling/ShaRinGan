#!/bin/bash

function cvt()
{
  CKPT=$1
  python sharingan_cvt.py \
    --output_dir "../data/cvt/pretrained_model" \
    --input_dir "../data/input/evaluation" \
    --max_steps 100000 \
    --checkpoint ../data/output/pretrained_model/$CKPT
}

CKPTS=`ls ../data/output/pretrained_model | grep -Po "model\.ckpt-\d+" | uniq`
for i in $CKPTS ; do 
  cvt $i
done
