#!/bin/bash

function train () {
  EPOCHS=$1
  LR=$2
  BETA1=$3
  STD=$4
  BATCH_SIZE=$5
  NGF=32
  NDF=32

  echo EPOCHS=${EPOCHS}
  echo LR=${LR}
  echo BETA1=${BETA1}
  echo STD=${STD}
  echo BATCH_SIZE=${BATCH_SIZE}
  echo NGF=${NGF}
  echo NDF=${NDF}
  
  SUFFIX=lr${LR}_b1${BETA1}_std${STD}_bs${BATCH_SIZE}
  OUT_DIR="cp_${SUFFIX}"

  echo removing ${OUT_DIR}
  rm -rf ${OUT_DIR}
  echo creating ${OUT_DIR}
  mkdir -p ${OUT_DIR}

  python sharingan_train.py \
    --output_dir "${OUT_DIR}" \
    --max_epochs ${EPOCHS} \
    --lr ${LR} \
    --beta1 ${BETA1} \
    --conv_std ${STD} \
    --ngf ${NGF} \
    --ndf ${NGF} \
    --batch_size ${BATCH_SIZE} \
    --input_dir "../data/input/training" 2>&1 | tee "${OUT_DIR}/output.log"
}

train 200 0.00004 0.5 0.0005 32
train 200 0.0002 0.5 0.0005 32
train 200 0.001 0.5 0.0005 32

train 200 0.00004 0.5 0.0005 64
train 200 0.0002 0.5 0.0005 64
train 200 0.001 0.5 0.0005 64

train 200 0.00004 0.5 0.005 128
train 200 0.0002 0.5 0.005 128
train 200 0.001 0.5 0.005 128

train 200 0.00004 0.5 0.05 256
train 200 0.0002 0.5 0.05 256
train 200 0.001 0.5 0.05 256
