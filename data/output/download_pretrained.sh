#!/bin/bash

curl -L -O https://github.com/hamling-ling/ShaRinGan/releases/download/pretrained1.3/pretrained_model-1.3.zip
curl -L -O https://github.com/hamling-ling/ShaRinGan/releases/download/pretrained1.2/pretrained_model-1.2-quantized.zip

#for non-quantized model
unzip pretrained_model-1.3.zip

#for quantization aware pretrained model. Use this instead of above
#unzip pretrained_model-1.2-quantized.zip
