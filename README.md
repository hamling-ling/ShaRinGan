# ShaRinGan

DCGAN that converts raw guitar sound into effector modified sound.

## Getting Started

Clone this repository
```
git clone https://github.com/hamling-ling/ShaRinGan.git
```
 
Download training data set. This takes about 30 sec.
```
$cd ShaRinGan/data/raw_waves
$./download.sh
```

Convert wave files to bin files
```
$cd ../../src
$python create_waves_wavfiles.py
```

Run Training(you can skip)
```
$cd ../../src
# this takes a few hours with GPU
$python sharingan_train.sh
```

Or Download pre-trained model
```
$cd ../../data/output
$download_pretrained.sh
```

Run Inference to convert raw to effectored suond
```
$cd src
$./sharingan_cvt.sh
```

You will see raw input, ground truth and converted wav files
```
$ls ../data/cvt/pretrained_model/
input.wav  output.wav  target.wav
```

### Prerequisites

- tensorflow 1.4.0
- pysoundfile

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details


