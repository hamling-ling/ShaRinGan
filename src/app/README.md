# ShaRinGan Application

DCGAN guitar effector application for Jetson.
The model is based on [ShaRinGan](https://github.com/hamling-ling/ShaRinGan "ShaRinGan")

[![Watch the video](https://img.youtube.com/vi/b-zGMJ6IPrw/hqdefault.jpg)](https://youtu.be/b-zGMJ6IPrw)

## Overview

This project is about making a python app to turn the Jetson Nano into guitar effector. The Jetson-Nano captures guitar sound through audio interface.
Then it moduletes the sound like effector as a result of DC-GAN inference.

```
+--------+     +-------------+     +---------+
|        |     | Roland      |---->| Jetson  |
| Guitar |---->| DUO-CAPTURE |     | Nano    |
|        |     | Mk2         |<----|         |
+--------+     +------+------+     +---------+
                      |
+---------+           |
|         |           |
| Speaker |<----------+
|         |
+---------+
```

Roland DUO-CAPTURE mk2 is an audio interface. Connecting
it with USB cable makes Jetson-Nano be able to capture
a guitar sound. It also works as audio output interface.

## Requirements

- Jetson Nano
- JetPack 4.4
- Tensorflow 1.15.3
- pyaudio
- Audio interface

## Getting Started

### Setup Jetson Nano

1. Install JetPack 4.4 SD card image from [JetPack SDK]( https://developer.nvidia.com/embedded/jetpack "JetPack SDK").
1. Login to you Jetson Nano and proceed following commands.\
   (See [NVIDIA Documentation - Installing TensorFlow For Jetson Platform](https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html "NVIDIA Documentation - Installing TensorFlow For Jetson Platform") )
```
$ sudo apt-get update
$ sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran
$ sudo apt-get install python3-pip
$ sudo pip3 install -U pip testresources setuptools
$ sudo pip3 install -U numpy==1.16.1 future==0.18.2 mock==3.0.5 h5py==2.10.0 keras_preprocessing==1.1.1 keras_applications==1.0.8 gast==0.2.2 futures protobuf pybind11
```
3. Install pycuda\
  Add path to cuda
```
$ vi ~/.bashrc
```
Then add followings
```
export CPATH=$CPATH:/usr/local/cuda/include
export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda/lib64
```
Apply the change
```
$ source ~/.bashrc
```
Install pycuda
```
$ sudo pip3 install -U pycuda
```
4. Install Tensorflow 1.15.3
```
$ sudo pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v44 tensorflow==1.15.3+nv20.7
```
5. Setup pyaudio
```
$ sudo apt-get install portaudio19-dev
$ sudo pip3 install pyaudio
```

### Run the App
1. Clone this repository
```
$ git clone https://github.com/hamling-ling/ShaRinGan.git
```
2. Go to app directory
```
$ cd ShaRinGan/src/app
```
3. Download model file
```
$ ./download_model.sh
```
5. Confirm audio interface\
If you are using an audio interface other than DUO-CAPTURE,
you need to modify following line of sharingan.py
```
DEVICE_NAME = 'DUO-CAPTURE'
```
6. Setup DUO-CAPTURE mk2 (If you are using)\
  You need to set EXT switch to "**" not "*" (as described [here]( https://ubuntuforums.org/showthread.php?t=1905531 "Ubuntu forums - Roland USB audio interface impossible to make it work"))
4. Run the app
```
$ ./sharingan.sh
```

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
