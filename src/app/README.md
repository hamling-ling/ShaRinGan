# ShaRinGan Application

DCGAN guitar effector application for Jetson.
The model is based on [ShaRinGan](https://github.com/hamling-ling/ShaRinGan "ShaRinGan")

[![Watch the video](https://img.youtube.com/vi/b-zGMJ6IPrw/hqdefault.jpg)](https://youtu.be/b-zGMJ6IPrw)

## Overview

This project is about making a python app to turn the Jetson Nano into guitar effector. The Jetson-Nano captures guitar sound through audio interface.
Then it moduletes the sound like effector as a result of by DC-GAN inference.

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
- JetPack 4.2.3
- Tensorflow 1.13.1
- pyaudio
- Audio interface

## Getting Started

### Setup Jetson Nano
1. Install JetPack 4.2.3 SD Card Image from [JetPack Archive]( https://developer.nvidia.com/embedded/jetpack-archive "JetPack Archive").
2. Login to you Jetson Nano and proceed following commands.
   (See [NVIDIA Documentation - Installing TensorFlow For Jetson Platform](https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html "NVIDIA Documentation - Installing TensorFlow For Jetson Platform") )
```
sudo apt update
sudo apt install python3-pip libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran
sudo pip3 install -U pip testresources setuptools

```
3. Install pycuda
```
```
3. Install Tensorflow 1.13.1
```
sudo pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v42 tensorflow-gpu==1.13.1+nv19.3
```
4. Setup pyaudio
```
sudo apt-get install portaudio19-dev
pip3 install pyaudio
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
5. Confirm audio interface.
   If you are using an audio interface other than DUO-CAPTURE,
   you need to modify following line of sharingan.py
```
DEVICE_NAME = 'DUO-CAPTURE'
```
6. Setup DUO-CAPTURE mk2 (If you are using)
   You need to set EXT switch to "**" not "*" (as described [here]( https://ubuntuforums.org/showthread.php?t=1905531 "Ubuntu forums - Roland USB audio interface impossible to make it work"))
4. Run the app
```
$ ./sharingan.sh
```

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
