import pyaudio
import time
import threading
import os
import numpy as np
import audio_streamer
import effector

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

PB_FILE = '../../data/output/frozen_model/frozen.pb'
UFF_FILE = '../../data/output/frozen_model/frozen.uff'
ENGINE_FILE = '../../data/output/frozen_model/tensorrt_engine_fp'
DEVICE_NAME = 'DUO-CAPTURE'

"""
Provides guitor effector functionality using AI
"""
class Sharingan():
    def __init__(self):
        self.event = threading.Event()
        self.input = np.zeros([1,1,1024])
        self.output = np.zeros([1,1,1024])
        self.effector = effector.Effector(PB_FILE, UFF_FILE, ENGINE_FILE)

    def start(self):
        """
        start sharingan. has to be run from main thread
        since I can't touch cuda from other thread somehow.
        """
        self.effector.create_engine()
        self.effector.initialize_engine()

        audio = audio_streamer.AudioStreamer(input_device_name=DEVICE_NAME, output_device_name=DEVICE_NAME)
        audio.open_device(self, self.audio_arrived)
        audio.start_streaming()

        while audio.is_streaming():
            # Wait to audio data arrived
            self.event.wait()
            # Once audio data is arrived. is should be stored in self.input
            # We give the data to effector to get modified sound
            # Send signal to audio callback function
            self.output = self.effector.effect(self.input)
            self.event.set()
            self.event.clear()

        audio.stop_streaming()
        audio.close_device()

    def audio_arrived(self, context, seq, in_data):
        # Store data and activate inference process
        self.input = in_data
        self.event.set()
        self.event.clear()
        # Sait until inference is done
        self.event.wait()

        # return data to play
        return self.output

if __name__ == "__main__":
    sharingan = Sharingan()
    sharingan.start()
