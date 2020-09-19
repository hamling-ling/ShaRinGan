import pyaudio
import time
import numpy as np
import audio_utility as au

class AudioStreamer():
    def __init__(self, input_device_name, output_device_name):
        self.input_device_name = input_device_name
        self.output_device_name = output_device_name
        self.channels = 1 # mono micriphone
        self.rate = 44100 # CD quality
        self.format = pyaudio.paInt16
        self.seq = 0

    def get_inout_devices(self):
        input_device = None
        output_device = None
        retry_counter = 0
        while retry_counter < 10:
            input_device = au.get_pyaudio_device(self.p, self.input_device_name)
            output_device = au.get_pyaudio_device(self.p, self.output_device_name)
            if(input_device is not None and output_device is not None):
                break
            if(input_device is None):
                print("retrying to get audio input device", self.input_device_name)
            if(output_device is None):
                print("retrying to gete audio output device", self.output_device_name)

            # Re-create pyaudio and try again
            self.p.terminate()
            self.p = pyaudio.PyAudio()
            time.sleep(1)
            retry_counter = retry_counter + 1
        return input_device, output_device

    def open_device(self, callback_context, callback):
        self.p = pyaudio.PyAudio()
        input_device, output_device = self.get_inout_devices()
        if(input_device is None):
            msg = "input device {0} not found".format(self.input_device_name)
            self.p.terminate()
            raise ValueError(msg)
        if(output_device is None):
            msg = "output device {0} not found".format(self.output_device_name)
            self.p.terminate()
            raise ValueError(msg)

        self.user_callback = callback
        self.user_context = callback_context
        self.stream = self.p.open(
                        input_device_index=input_device.get('index'),
                        output_device_index=output_device.get('index'),
                        format=self.format,
                        channels=self.channels,
                        rate=self.rate,
                        frames_per_buffer=1024,
                        output=True,
                        input=True,
                        stream_callback=self.data_arrived,
                        start=False )
        print(self.input_device_name, " opend for input")
        print(self.output_device_name, " opend for output")

    def close_device(self):
        self.callback = None
        self.stream.close()
        self.p.terminate()
        self.stream = None
        self.p = None

    def start_streaming(self):
        self.stream.start_stream()

    def stop_streaming(self):
        self.stream.stop_stream()

    def is_streaming(self):
        return self.stream.is_active()

    def data_arrived(self, in_data, frame_count, time_info, status):
        # convert binary array to int16, then normalize to float
        in_floats=np.frombuffer(in_data, dtype="int16")/np.float32(32768.0)
    
        # callback and receive output data
        start_time = time.time()
        out_floats = self.user_callback(self.user_context, self.seq, in_floats)
        milli_sec = ((time.time()-start_time)*1000)
        if(22.0 < milli_sec):
            print("took ", milli_sec, "ms might be dropping frame data")
        self.seq = self.seq + 1

        # convert returned data from callback to pyaudio data
        denorm=out_floats*32768
        out_data16 = denorm.astype(np.int16)
        out_data = out_data16.tobytes()
    
        return (out_data, pyaudio.paContinue)

    def close(self):
        self.p.terminate()
