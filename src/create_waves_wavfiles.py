import numpy as np
import soundfile as sf
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os

sz=1024
offsets = np.array([0])

training_fn_pairs = [
    ["../data/raw_waves/gt_notes.wav", "../data/raw_waves/gt_notes_dist.wav"],
    ["../data/raw_waves/gt_many_notes.wav", "../data/raw_waves/gt_many_notes_dist.wav"]
]

evaluation_fn_pairs = [
    ["../data/raw_waves/vivaldi_spring.wav", "../data/raw_waves/vivaldi_spring_dist.wav"]
]

def readWave(path_src, path_cnv):
    data_src, samplerate_src = sf.read(path_src)
    data_cnv, samplerate_cnv = sf.read(path_cnv)
    data_len = min(data_src.shape[0], data_cnv.shape[0])
    print("read ", path_src, " and ", path_cnv)
    print("len=", data_len)
    data_src = np.array(data_src)
    data_cnv = np.array(data_cnv)
    data_src_trim = data_src[:data_len]
    data_cnv_trim = data_cnv[:data_len]
    return data_src_trim, data_cnv_trim

def outputWave(input_fn_pairs, output_path, is_training):
    file_counter = 0
    for fnp in input_fn_pairs:
        data_src, data_cnv  = readWave(fnp[0], fnp[1])
        loops = int(len(data_src)/sz) - 1
        for i in np.arange(loops):
            for j in np.arange(offsets.shape[0]):
                data = np.zeros([2,sz], dtype=np.float32)
                i_from = i * sz + offsets[j]
                data[0] = data_src[i_from:i_from+sz]
                data[1] = data_cnv[i_from:i_from+sz]
                fn = os.path.join(output_path, "{0:0>6}.bin".format(file_counter))
                data.tofile(fn)
                print(fn, " saved")
                file_counter = file_counter + 1
                if not is_training:
                    break

out_train="../data/input/training"
out_validation = "../data/input/validation"
out_evaluation = "../data/input/evaluation"

os.makedirs("../data", exist_ok=True)
os.makedirs("../data/input", exist_ok=True)
os.makedirs(out_train, exist_ok=True)
os.makedirs(out_validation, exist_ok=True)
os.makedirs(out_evaluation, exist_ok=True)

outputWave(training_fn_pairs, out_train, True)
outputWave(evaluation_fn_pairs, out_evaluation, False)
