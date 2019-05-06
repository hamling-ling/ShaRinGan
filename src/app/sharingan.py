import pyaudio
import time
import threading
import os
import sys
import select
import numpy as np
import audio_streamer
import effector
import graceful_killer as kl

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

PB_FILE = './frozen_model/frozen.pb'
UFF_FILE = './frozen_model/frozen.uff'
ENGINE_FILE = './tensorrt_engine_fp'
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
        self.enabled = True
        self.quit = False
        self.killer = kl.GracefulKiller()

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

        self.print_instruction()
    
        while audio.is_streaming() and not self.should_quit():
            # Wait to audio data arrived
            self.event.wait()
            # Once audio data is arrived. is should be stored in self.input
            # We give the data to effector to get modified sound
            # Send signal to audio callback function
            if(self.enabled):
                self.output = self.effector.effect(self.input)
            else:
                self.output = self.input
            self.event.set()
            self.event.clear()
            # Receive Keyboard iput
            self.handle_input()

        print("quitting")
        self.event.set()
        self.event.clear()
        audio.stop_streaming()
        audio.close_device()
        print("sharingan finished")

    def audio_arrived(self, context, seq, in_data):
        if(self.quit):
            return in_data
        # Store data and activate inference process
        self.input = in_data
        self.event.set()
        self.event.clear()
        # Sait until inference is done
        self.event.wait()

        # return data to play
        return self.output

    def should_quit(self):
        if(self.quit):
            return True
        if(self.killer.kill_now):
            return True
        return False

    def print_instruction(self):
        print("input 'q' to quit")
        print("      'e' to enable effector(default)")
        print("      'd' to disable effector")

    def handle_input(self):
        ch = None
        if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
            ch = sys.stdin.read(1)
        if ch is None:
            return
        elif ch == 'q' or ch == 'Q':
            print("quit")
            self.quit = True
        elif ch == 'e' or ch == 'E':
            self.enabled = True
            print("effector enabled")
        elif ch == 'd' or ch == 'd':
            self.enabled = False
            print("effector disabled")

if __name__ == "__main__":
    sharingan = Sharingan()
    sharingan.start()
