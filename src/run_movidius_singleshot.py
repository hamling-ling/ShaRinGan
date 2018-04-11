from mvnc import mvncapi as mvnc
import sys
import os
import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

path_to_networks = './'
path_to_images = '../data/input/evaluation/'
graph_filename = 'graph'
image_filename = path_to_images + '*.bin'

mvnc.SetGlobalOption(mvnc.GlobalOption.LOGLEVEL, 2)

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
with open(path_to_networks + graph_filename, mode='rb') as f:
    graphfile = f.read()
graph = device.AllocateGraph(graphfile)

#Load data
inputs = np.fromfile(image_filename, dtype=np.float32)
inputs = inputs.astype(np.float16) # cast
inputs = np.reshape(inputs, [2,1024])

print('Start download to NCS...')
data_in = np.reshape(inputs[0], [1,1,1024,1])
print(data_in.shape)
graph.LoadTensor(data_in, 'input')

output, userobj = graph.GetResult() #this will timeout! why?
graph.DeallocateGraph()
device.CloseDevice()

print("output.shape=", output.shape)

data_plt_in = data_in
data_plt_out = np.reshape(output, [1,1,1024,1])
data_plt_tgt = np.reshape(inputs[1], [1,1,1024,1])
save_plots(data_plt_in, data_plt_out, data_plt_tgt, "movidius.png")

print('Finished')
