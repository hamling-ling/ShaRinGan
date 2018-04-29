from mvnc import mvncapi as mvnc
import sys
#import os
import numpy as np
import tensorflow as tf

path_to_networks = './'
graph_filename = 'graph'

mvnc.SetGlobalOption(mvnc.GlobalOption.LOG_LEVEL, 2)

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
data_in = np.ones([1, 1, 1024, 1])

print('Start download to NCS...')
graph.LoadTensor(data_in, 'input')

output = None
try:
    output, userobj = graph.GetResult() #this will timeout! why?
except Exception as e:
    print(e)
finally:
    graph.DeallocateGraph()
    device.CloseDevice()

if output is not None:
    print("output.shape=", output.shape)

print('Finished')
