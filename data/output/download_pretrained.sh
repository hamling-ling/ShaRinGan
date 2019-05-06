#!/bin/bash

curl -L -o pretrained_model-1.3.zip "https://docs.google.com/uc?export=download&id=1k-Mpsnrn5SC8zvVsT-Q2wjClprQspa-0"
curl -L -O https://github.com/hamling-ling/ShaRinGan/releases/download/pretrained1.2/pretrained_model-1.2-quantized.zip

#for non-quantized model
unzip pretrained_model-1.3.zip

#for quantization aware pretrained model. Use this instead of above
#unzip pretrained_model-1.2-quantized.zip
