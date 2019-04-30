import os
import time
import numpy as np
import tensorrt as trt
import uff
import pycuda.driver as cuda
import pycuda.autoinit

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class Effector():
    def __init__(self, pb_file_name, uff_file_name, engine_file_name):
        print("creating effector object")
        self.pb_file_name = pb_file_name
        self.uff_file_name = uff_file_name
        self.engine_file_name = engine_file_name
        print("effector object created")

    def create_engine(self):
        print("creating engine")
        if not os.path.exists(self.uff_file_name):
            print(self.uff_file_name, "not found. create new uff file")
            self.convert_to_uff()

        if not os.path.exists(self.engine_file_name):
            print(self.engine_file_name, "not found. build new engine")
            engine = self.build_engine()
            with open(self.engine_file_name, "wb") as f:
                f.write(engine.serialize())
        else:
            print(self.engine_file_name, "found. reuse engine")
            with open(self.engine_file_name, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
                engine = runtime.deserialize_cuda_engine(f.read())
        self.engine = engine
        print("engine created")
    
    def initialize_engine(self):
        print("initializing engine")
        self.h_input  = cuda.pagelocked_empty(
            trt.volume(self.engine.get_binding_shape(0)),
            dtype=np.float32)
        self.h_output = cuda.pagelocked_empty(
            trt.volume(self.engine.get_binding_shape(1)),
            dtype=np.float32)
        self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        self.d_output = cuda.mem_alloc(self.h_output.nbytes)
        self.stream = cuda.Stream()
        self.execution_context = self.engine.create_execution_context()
        print("engine initialized")

    def effect(self, input):
        np.copyto(self.h_input, input)
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)

        self.execution_context.execute_async(
            bindings=[int(self.d_input),
            int(self.d_output)],
            stream_handle=self.stream.handle)

        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        self.stream.synchronize()
        return self.h_output

    def build_engine(self):
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.UffParser() as parser:
            parser.register_input("input", (1, 1, 1024))
            parser.register_output("generator/Tanh")
            parser.parse(self.uff_file_name, network)
        
            builder.max_workspace_size = 1 <<  20
            builder.fp16_mode = True
            builder.strict_type_constraints = True

            engine = builder.build_cuda_engine(network)
            return engine

    def convert_to_uff(self):
            uff.from_tensorflow_frozen_model(frozen_file=self.pb_file_name,
                                        output_nodes=["generator/Tanh"],
                                        output_filename=self.uff_file_name,
                                        list_nodes=False)