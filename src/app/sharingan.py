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

PB_FILE     = './frozen_model/frozen.pb'
UFF_FILE    = './frozen_model/frozen.uff'
ENGINE_FILE = './tensorrt_engine_fp'
DEVICE_NAME = 'DUO-CAPTURE'

"""
Provides guitor effector functionality using AI
"""
class Sharingan():

    TIMEOUT_SEC = 0.5*1024.0*1.0/44100.0

    def __init__(self):
        self.input = np.zeros([1,1,1024])
        self.output = np.zeros([1,1,1024])
        self.effector = effector.Effector(PB_FILE, UFF_FILE, ENGINE_FILE)
        self.enabled = True
        self.quit = False
        self.input_lock = threading.Lock()
        self.output_condition = threading.Condition(threading.Lock())
        self.output_condition.state = False # True: notify called, otherwise False
        self.killer = kl.GracefulKiller()
        
    def start(self):
        """
        start sharingan. has to be run from main thread
        since I can't touch cuda from other thread somehow.
        """
        self.effector.create_engine()
        self.effector.initialize_engine()

        # warm up
        for i in range(10):
            self.effector.effect(np.zeros([1, 1, 1024]))
        
        audio = audio_streamer.AudioStreamer(input_device_name=DEVICE_NAME, output_device_name=DEVICE_NAME)
        audio.open_device(self, self.audio_arrived)
        audio.start_streaming()

        self.print_instruction()

        while audio.is_streaming() and not self.should_quit():

            # Once audio data is arrived. is should be stored in self.input
            self.input_lock.acquire()
            local_input = None
            if(self.input is not None):
                local_input = np.copy(self.input)
            self.input_lock.release()

            if(local_input is not None):
                # We give the data to effector to modulate sound data
                # Send signal to audio callback function
                if(self.enabled):
                    local_output = self.effector.effect(local_input)
                else:
                    local_output = local_input

                self.output_condition.acquire()
                self.output_condition.state = False

                self.output = np.copy(local_output)

            self.output_condition.state = True
            self.output_condition.notify()
            self.output_condition.release()

            # Receive Keyboard iput
            self.handle_input()

        print("quitting")
        audio.stop_streaming()
        audio.close_device()
        print("sharingan finished")

    def audio_arrived(self, context, seq, in_data):
        if(self.quit):
            return in_data

        self.input_lock.acquire()
        # store data that will be processed by main thread
        context.input = np.copy(in_data)
        self.input_lock.release()

        self.output_condition.acquire()
        self.output_condition.state = False
        # wait for main thread finish processing
        self.output_condition.wait(self.TIMEOUT_SEC)

        # if timeout, use in_data to play, otherwise use processed data
        is_timeout   = self.output_condition.state == False
        final_output = None
        if(is_timeout):
            print("timeout")
            final_output = in_data
        else:
            final_output = np.copy(context.output)

        self.output_condition.release()

        # return data to play
        return final_output

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
