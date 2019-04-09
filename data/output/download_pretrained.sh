#!/bin/bash

curl -L -O https://github.com/hamling-ling/ShaRinGan/releases/download/pretrained1.2/pretrained_model-1.2.zip
curl -L -O https://github.com/hamling-ling/ShaRinGan/releases/download/pretrained1.2/pretrained_model-1.2-quantized.zip

unzip pretrained_model-1.2.zip

#for quantization aware pretrained model. Use this instead of above
#unzip pretrained_model-1.2-quantized.zip
