from mvnc import mvncapi as mvnc
import sys
import argparse
import os
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

mvnc.global_set_option(mvnc.GlobalOption.RW_LOG_LEVEL, mvnc.LogLevel.DEBUG)

def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", required=True, help="graph file")
    parser.add_argument("--input", required=True, help="path to a input bin file")
    parser.add_argument("--output_dir", required=True, help="where to put output file")

    return parser.parse_args()

def load_input_tensor(file):
    #Load data
    inputs = np.fromfile(file, dtype=np.float32)
    inputs = np.reshape(inputs, [2,1024])

    print('Start download to NCS...')
    tensor = np.reshape(inputs[0], [1,1,1024,1])
    print(tensor.shape)

    target = np.reshape(inputs[1], [1,1,1024,1])
    return tensor, target

def get_device():
    devices = mvnc.enumerate_devices()
    if len(devices) == 0:
        print('No devices found')
        quit()

    device = mvnc.Device(devices[0])
    return device

def save_plots(input, output, target, filename):

    plt.clf()
    plt.plot(input[0,0,:,0],  linestyle='solid',  label='input')
    plt.plot(output[0,0,:,0], linestyle='dashed', label='output')
    plt.plot(target[0,0,:,0], linestyle='dotted', label='target')
    plt.legend()
    plt.savefig(filename)
    plt.clf()
    print(filename, " saved")

def save_result(output_path, data_in, data_tgt, data_out):
    data_plt_in  = np.reshape(data_in,  [1,1,1024,1])
    data_plt_tgt = np.reshape(data_tgt, [1,1,1024,1])
    data_plt_out = np.reshape(data_out, [1,1,1024,1])

    os.makedirs(output_path, exist_ok=True)
    out_image_file = os.path.join(output_path, "movidius.png")
    save_plots(data_plt_in, data_plt_out, data_plt_tgt, out_image_file)

def main():
    a = process_args()
    tensor, target = load_input_tensor(a.input)
    device = get_device()

    is_inferenced = False
    output = None
    try:
        device.open()

        #Load graph
        with open(a.graph, mode='rb') as f:
            graphfile_buffer = f.read()

        graph = mvnc.Graph('graph1')
        input_fifo, output_fifo = graph.allocate_with_fifos(device, graphfile_buffer)

        #Enqueu data
        graph.queue_inference_with_fifo_elem(   input_fifo,
                                                output_fifo,
                                                tensor.astype(np.float32),
                                                'user object')
        #Get Result
        output, userobj = output_fifo.read_elem()
        is_inferenced = True
    except Exception as e:
        print(e)
    finally:
        if(input_fifo):
            input_fifo.destroy()
        if(output_fifo):
            output_fifo.destroy()
        if(graph):
            graph.destroy()
        device.close()
        device.destroy()

    if(is_inferenced):
        print("output.shape=", output.shape)
        save_result(a.output_dir, tensor, target, output)

main()

print('Finished')
