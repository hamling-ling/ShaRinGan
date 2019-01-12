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
from collections import namedtuple

def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", help="path to folder containing images")
    parser.add_argument("--output_dir", required=True, help="where to put output files")
    parser.add_argument("--checkpoint", required=True, help="checkpoint directory or ckpt path (ex sharingan_checkpoints/model.ckpt-1001) to use for testing")
    parser.add_argument("--max_steps", type=int, help="number of training steps (0 to disable)")
    parser.add_argument("--batch_size", type=int, default=1, help="number of images in batch")

    a = parser.parse_args()

    if(os.path.isdir(a.checkpoint)):
        dir_cp = a.checkpoint
    else:
        dir_cp = os.path.dirname(a.checkpoint)

    dir_cp = os.path.dirname(a.checkpoint)
    filename = os.path.join(dir_cp, "hyper_params.json")
    with open(filename) as fd:
        json_str = fd.read()
        print("hyper parameters=\n", json_str)
        hyp = json.loads(json_str, object_hook=lambda d: namedtuple('HyperParams', d.keys())(*d.values()))

    return a, hyp


def main():
    a, hyper_params = process_args()

    examples = load_examples(input_dir=a.input_dir, batch_size=a.batch_size, is_training=False)
    print("examples count = %d" % examples.count)

    # inputs and targets are [batch_size, height, width, channels]
    with tf.variable_scope("generator"):
        model = create_generator(generator_inputs           = examples.inputs,
                                generator_outputs_channels = 1,
                                ngf                        = hyper_params.ngf,
                                is_training                = False,
                                is_fused                   = False)

    inputs = examples.inputs
    targets = examples.targets
    outputs = model

    with tf.name_scope("encode_images"):
        display_fetches = {
            "paths": examples.paths,
            "inputs": inputs,
            "targets": targets,
            "outputs": outputs,
        }

    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    server = tf.train.Server.create_local_server()
    saver = tf.train.Saver()

    max_steps = a.max_steps
    if(max_steps is None):
        max_steps = examples.steps_per_epoch

    init_op=tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    scaffold = tf.train.Scaffold(init_op)

    with tf.train.MonitoredTrainingSession(master=server.target,
                                       config=tf.ConfigProto(allow_soft_placement=True),
                                       is_chief=True,
                                       scaffold = scaffold) as sess:
        print("loading model from checkpoint")
        if os.path.isdir(a.checkpoint):
            print("restoring latest in ", a.checkpoint)
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            saver.restore(sess, checkpoint)
        else:
            print("restoring from a file ", a.checkpoint)
            saver.restore(sess, a.checkpoint)
        coord = tf.train.Coordinator()

        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        wave_in = []
        wave_tgt = []
        wave_out = []

        try:
            counter = 0
            while not sess.should_stop():
                if(max_steps < counter):
                    break

                fetches = sess.run(display_fetches)
                inputs=fetches["inputs"]
                targets=fetches["targets"]
                outputs=fetches["outputs"]

                wave_in.extend(inputs[0,0,:,0].tolist())
                wave_tgt.extend(targets[0,0,:,0].tolist())
                wave_out.extend(outputs[0,0,:,0].tolist())
                print("{0}/{1}".format(counter, examples.count))

                counter = counter + 1
        except tf.errors.OutOfRangeError:
            print('Done converting -- epoch limit reached')
        finally:
            coord.request_stop()
            coord.join(threads)

        fn_input = os.path.join(a.output_dir, "input.wav")
        fn_target = os.path.join(a.output_dir, "target.wav")
        fn_output = os.path.join(a.output_dir, "output.wav")

        os.makedirs(a.output_dir, exist_ok=True)
        sf.write(fn_input, wave_in, 44100)
        print(fn_input, " saved")
        sf.write(fn_target, wave_tgt, 44100)
        print(fn_target, " saved")
        sf.write(fn_output, wave_out, 44100)
        print(fn_output, " saved")

main()
