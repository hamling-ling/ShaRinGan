import os
import time
import numpy as np
import tensorflow.compat.v1 as tf
import tensorrt as trt
import uff
import pycuda.driver as cuda
import pycuda.autoinit
import matplotlib.pyplot as plt


PB_FILE = '../data/output/frozen_model/frozen.pb'
UFF_FILE = '../data/output/frozen_model/frozen.uff'
ENGINE_FILE = '../data/output/frozen_model/tensorrt_engine_fp'
RESULT_PLOT_FILE = '../data/output/frozen_model/tensorrt_engine_fp.png'

tf.disable_v2_behavior()
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def convert_to_uff(pb_file_in, uff_file_out):
        uff.from_tensorflow_frozen_model(frozen_file=pb_file_in,
                                    output_nodes=["generator/Tanh"],
                                    output_filename=uff_file_out,
                                    list_nodes=False)

def build_engine(uff_file):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.UffParser() as parser:
        parser.register_input("input", (1, 1, 1024))
        parser.register_output("generator/Tanh")
        parser.parse(uff_file, network)
        
        builder.max_workspace_size = 1 <<  20
        builder.fp16_mode = True
        builder.strict_type_constraints = True

        engine = builder.build_cuda_engine(network)
        return engine

def inference(engine, input):
    # Determine dimensions and create page-locked memory buffers (i.e. won't be swapped to disk) to hold host inputs/outputs.
    h_input  = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=np.float32)
    h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=np.float32)
    # Allocate device memory for inputs and outputs.
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    # Create a stream in which to copy inputs/outputs and run inference.
    stream = cuda.Stream()
    
    with engine.create_execution_context() as context:
        start_time = time.time()
        np.copyto(h_input, input)
        # Transfer input data to the GPU.
        cuda.memcpy_htod_async(d_input, h_input, stream)
        # Run inference.
        context.execute_async(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        # Synchronize the stream
        stream.synchronize()
        print('time: %fms'%((time.time()-start_time)*1000))
        # Return the host output. 
        return h_output

def load_data(file_name):
    raw_data=np.fromfile(file_name, np.float32)
    raw_data_reshaped = np.reshape(raw_data,[2,1,-1,1])
    input  = raw_data_reshaped[0].reshape([1,1,1024])
    ground_truth = raw_data_reshaped[1].reshape([1,1,1024])
    return input, ground_truth

def save_plot(input, output, ground_truth, filename):

    plt.clf()

    plt.plot(input.flatten(), linestyle='solid')
    plt.plot(ground_truth.flatten(), linestyle='dotted')
    plt.plot(output.flatten(), linestyle='dashed')
    
    plt.savefig(filename)
    plt.clf()
    print(filename, " saved")

def main():
    if not os.path.exists(UFF_FILE):
        convert_to_uff(PB_FILE, UFF_FILE)

    if not os.path.exists(ENGINE_FILE):
        engine = build_engine(UFF_FILE)
        with open(ENGINE_FILE, "wb") as f:
            f.write(engine.serialize())
    else:
        with open(ENGINE_FILE, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())

    # load input and ground truth
    fn = "../data/input/evaluation/002080.bin"
    input, ground_truth = load_data(fn)

    # perform inference
    output = inference(engine, input)

    # create visualized result
    save_plot(input, output, ground_truth, RESULT_PLOT_FILE)

main()
