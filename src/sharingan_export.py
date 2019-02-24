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
from collections import namedtuple

SZ = 1024

def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True, help="where to put output files")
    parser.add_argument("--checkpoint", required=True, help="checkpoint directory or ckpt path (ex sharingan_checkpoints/model.ckpt-1001) to use for testing")

    a = parser.parse_args()

    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)

    if(os.path.isdir(a.checkpoint)):
        dir_cp = a.checkpoint
    else:
        dir_cp = os.path.dirname(a.checkpoint)

    filename = os.path.join(dir_cp, "hyper_params.json")
    with open(filename) as fd:
        json_str = fd.read()
        print("hyper parameters=\n", json_str)
        hyp = json.loads(json_str, object_hook=lambda d: namedtuple('HyperParams', d.keys())(*d.values()))

    return a, hyp

def save_last_node_name(node_name):
    f = open('last_node_name.txt', 'w')
    f.write(node_name)
    f.close()
    print(node_name, " saved in last_node_name.txt")

def main():
    a, hyper_params = process_args()

    tf.reset_default_graph()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        input = tf.placeholder("float", [1, 1, SZ, 1], name="input")
        with tf.variable_scope("generator"):
            generator = create_generator(generator_inputs           = input,
                                         generator_outputs_channels = 1,
                                         ngf                        = hyper_params.ngf,
                                         is_training                = False,
                                         is_fused                   = False)
            last_node_name=generator.name.split(":")[0]
            save_last_node_name(generator.name.split(":")[0])

        saver = tf.train.Saver(tf.global_variables())
        print("saver created")

        print("loading model from checkpoint")
        if os.path.isdir(a.checkpoint):
            print("restoring latest in ", a.checkpoint)
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            saver.restore(sess, checkpoint)
        else:
            print("restoring from a file ", a.checkpoint)
            saver.restore(sess, a.checkpoint)
        print("restored")

        outfile = os.path.join(a.output_dir, "movidius")
        saver.save(sess, outfile)
        print("saved ", outfile)

        # We use a built-in TF helper to export variables to constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, # The session is used to retrieve the weights
            tf.get_default_graph().as_graph_def(), # The graph_def is used to retrieve the nodes 
            [last_node_name]
        ) 

    # Finally we serialize and dump the output graph to the filesystem
    frozen_model_name=os.path.join(a.output_dir, "frozen_model.pb")
    with tf.gfile.GFile(frozen_model_name, "wb") as f:
        f.write(output_graph_def.SerializeToString())
    print("%d ops in the final graph." % len(output_graph_def.node))

    # save
    tflite_name = os.path.join(a.output_dir, "model.tflite")
    converter = tf.contrib.lite.TFLiteConverter.from_frozen_graph(
        graph_def_file=frozen_model_name,
        input_arrays=["input"],
        output_arrays=["generator/Tanh"])
    tflite_model = converter.convert()
    open(tflite_name, "wb").write(tflite_model)

    col = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
    for item in col:
        print(item)
main()
