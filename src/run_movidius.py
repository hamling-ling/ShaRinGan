#python run_movidius.py --input_dir ../data/input/evaluation --output_dir ./ --graphfile ./graph

from mvnc import mvncapi as mvnc
import sys
import os
import numpy as np
import argparse
import glob
import soundfile as sf

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", required=True, help="path to folder containing bin files")
parser.add_argument("--output_dir", required=True, help="where to put output files")
parser.add_argument("--graphfile", required=True, help="where to put output files")

a = parser.parse_args()

filepaths = glob.glob(os.path.join(a.input_dir, '*.bin'))
filepaths.sort()

mvnc.SetGlobalOption(mvnc.GlobalOption.LOG_LEVEL, 2)

def save_plots(input, output, target, filename):

    plt.clf()
    plt.plot(input[0,0,:,0], linestyle='solid')
    plt.plot(output[0,0,:,0], linestyle='dashed')
    plt.plot(target[0,0,:,0], linestyle='dotted')

    plt.savefig(filename)
    plt.clf()
    print(filename, " saved")

devices = mvnc.EnumerateDevices()
if len(devices) == 0:
    print('No devices found')
    quit()

device = mvnc.Device(devices[0])
device.OpenDevice()

#Load graph
with open(a.graphfile, mode='rb') as f:
    graphfile = f.read()
graph = device.AllocateGraph(graphfile)

wave_out = []

for filename in filepaths:
    print("processing ", filename)
    #Load data
    inputs = np.fromfile(filename, dtype=np.float32)
    inputs = inputs.astype(np.float16) # cast
    inputs = np.reshape(inputs, [2,1024])

    data_in = np.reshape(inputs[0], [1,1,1024,1])
    graph.LoadTensor(data_in, 'input')

    output, userobj = graph.GetResult()

    wave_out.extend(output.tolist())

graph.DeallocateGraph()
device.CloseDevice()

fn_output = os.path.join(a.output_dir, "output.wav")
sf.write(fn_output, wave_out, 44100)
print(fn_output, " saved")


print('Finished')
