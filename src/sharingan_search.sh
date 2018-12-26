#!/bin/bash

function train () {
  EPOCHS=$1
  LR=$2
  NGF=$3
  BATCH_SIZE=$4

  echo EPOCHS=${EPOCHS}
  echo LR=${LR}
  echo NGF=${NGF}
  echo BATCH_SIZE=${BATCH_SIZE}

  SUFFIX=ngf${NGF}_lr${LR}_bs${BATCH_SIZE}
  OUT_DIR="cp_${SUFFIX}"

  echo removing ${OUT_DIR}
  rm -rf ${OUT_DIR}
  echo creating ${OUT_DIR}
  mkdir -p ${OUT_DIR}

  python sharingan_train.py \
    --output_dir "${OUT_DIR}" \
    --max_epochs 200 \
    --lr ${LR} \
    --ngf ${NGF} \
    --ndf ${NGF} \
    --batch_size ${BATCH_SIZE} \
    --input_dir "../data/input/training" 2>&1 | tee "${OUT_DIR}/output.log"
}

train 200 0.00004 16 64
train 200 0.0002 16 64
train 200 0.001 16 64

train 200 0.00004 16 512
train 200 0.0002 16 512
train 200 0.001 16 512

train 200 0.00004 16 1024
train 200 0.0002 16 1024
train 200 0.001 16 1024

train 200 0.00004 32 64
train 200 0.0002 32 64
train 200 0.001 32 64

train 200 0.00004 32 512
train 200 0.0002 32 512
train 200 0.001 32 512

train 200 0.00004 32 1024
train 200 0.0002 32 1024
train 200 0.001 32 1024
