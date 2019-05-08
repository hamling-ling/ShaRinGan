#!/bin/bash

function cvt()
{
  CP_NUM=$1
  python sharingan_cvt.py \
    --output_dir "../data/cvt/pretrained_model" \
    --input_dir "../data/input/evaluation" \
    --max_steps 100000 \
    --checkpoint ../data/output/pretrained_model/model.ckpt-$CP_NUM
}

cvt 5000
cvt 6000
cvt 7000
cvt 8000
cvt 9000

cvt 10000
cvt 11000
cvt 12000
cvt 13000
cvt 14000
cvt 15000
cvt 16000
cvt 17000
cvt 18000
cvt 19000

cvt 20000
cvt 21000
cvt 22000
cvt 23000
cvt 24000
cvt 25000
cvt 26000
cvt 27000
cvt 28000
cvt 29000

cvt 30000
cvt 31000
cvt 32000
cvt 33000
cvt 34000
cvt 35000
cvt 36000
cvt 37000
cvt 38000
cvt 39000

cvt 40000
cvt 41000
cvt 42000
cvt 43000
cvt 44000
cvt 45000
cvt 46000
cvt 47000
cvt 48000
cvt 49000

