#!/bin/bash

USER_OPT=
USER_PARAM=
if [ "$1" != "--root" ]; then
    USER_OPT=-u
    USER_PARAM=$(id -u $USER):$(id -g $USER)
fi

# https://docs.nvidia.com/deeplearning/sdk/tensorrt-container-release-notes/running.html
# do following once
#docker pull nvcr.io/nvidia/tensorflow:20.06-py3
# then
#nvidia-docker run -it --rm nvcr.io/nvidia/tensorrt:20.06-py3

function tfrt() {
    docker run --runtime=nvidia \
	   -v $HOME/Github:$HOME/Github -e "GRANT_SUDO=yes"\
	   $USER_OPT $USER_PARAM \
	   --rm \
	   -it \
	   -p 8889:8888 \
	   -p 6006:6006 \
	   tensor-rt bash
}

tfrt
