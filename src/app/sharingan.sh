#!/bin/bash

# execute this script under app directory
#     $ cd ShaRinGan/src/app
#     $ ./sharingan.sh

sudo nvpmodel -m 0
sudo jetson_clocks

python3 sharingan.py
