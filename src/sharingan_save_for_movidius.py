# mvNCCompile movidius.meta.meta -in=input -on generator/decoder_1/Tanh -s30

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os
import inspect
import json
from sharingan_base import *
import soundfile as sf

SZ = 1024

def processArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True, help="where to put output files")
    parser.add_argument("--checkpoint", required=True, help="directory with checkpoint to use for testing")
    a = parser.parse_args()

    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)

    return a


def main():

    a = processArgs()

    tf.reset_default_graph()

    with tf.Session() as sess:

        input = tf.placeholder("float", [1, 1, SZ, 1], name="input")
        with tf.variable_scope("generator"):
            generator = create_generator(input, 1, is_training=False, is_fused=False)

        print("loading model from checkpoint")
        print("checkpoint loaded")
        saver = tf.train.Saver(tf.global_variables())
        print("saver created")
        checkpoint = tf.train.latest_checkpoint(a.checkpoint)
        saver.restore(sess, checkpoint)
        print("restored")

        saver.save(sess, os.path.join(a.output_dir, "movidius.meta"))
        print("saved")

        col = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
        for item in col:
            print(item)
main()
