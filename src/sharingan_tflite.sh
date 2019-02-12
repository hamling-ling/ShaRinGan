#!/bin/bash

tflite_convert \
    --graph_def_file=../data/output/frozen_model/frozen_model.pb \
    --output_file=../data/output/frozen_model/model.tflite \
    --output_format=TFLITE \
    --input_shae=1,1,1024,1 \
    --input_array=input \
    --output_array=generator/Tanh \
    --inference_type=FLOAT \
    --input_data_type=FLOAT
