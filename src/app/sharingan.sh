#!/bin/bash

sudo nvpmodel -m 0
sudo jetson_clocks

python3 sharingan.py
