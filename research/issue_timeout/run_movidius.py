from mvnc import mvncapi as mvnc
import sys
#import os
import numpy as np
#import tensorflow as tf

path_to_networks = './'
graph_filename = 'graph'

#mvnc.SetGlobalOption(mvnc.GlobalOption.LOG_LEVEL, 2)
mvnc.global_set_option(mvnc.GlobalOption.RW_LOG_LEVEL, mvnc.LogLevel.DEBUG)

#devices = mvnc.EnumerateDevices()
devices = mvnc.enumerate_devices()
if len(devices) == 0:
    print('No devices found')
    quit()

device = mvnc.Device(devices[0])
device.open()

#Load graph
with open(path_to_networks + graph_filename, mode='rb') as f:
    graph_file_buffer = f.read()

graph = mvnc.Graph('graph1')
#graph = device.AllocateGraph(graphfile)
input_fifo, output_fifo = graph.allocate_with_fifos(device, graph_file_buffer)

#Load data
tensor = np.ones([1,1,1024, 1])

print('Start download to NCS...')
#graph.LoadTensor(data_in, 'input')
graph.queue_inference_with_fifo_elem(input_fifo, output_fifo, tensor.astype(np.float32), 'user object')

print("fifo created")

output = None
try:
    #output, userobj = graph.GetResult() #this will timeout! why?
    print("trying")
    output, user_obj = output_fifo.read_elem()
except Exception as e:
    print(e)
finally:
    #graph.DeallocateGraph()
    #device.CloseDevice()
    input_fifo.destroy()
    output_fifo.destroy()
    graph.destroy()
    device.close()
    device.destroy()

if output is not None:
    print("output.shape=", output.shape)

print('Finished')
