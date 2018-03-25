from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os
import inspect
import json
import random
from sharingan_base import *

def initRand():
    seed = 0
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def processArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", help="path to folder containing images")
    parser.add_argument("--output_dir", required=True, help="where to put output files")
    parser.add_argument("--checkpoint", default=None, help="directory with checkpoint to resume training from or use for testing")
    parser.add_argument("--max_steps", type=int, help="number of training steps (0 to disable)")
    parser.add_argument("--max_epochs", type=int, help="number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="number of images in batch")

    a = parser.parse_args()

    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)

    for k, v in a._get_kwargs():
        print(k, "=", v)

    with open(os.path.join(a.output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(a), sort_keys=True, indent=4))

    return a

def main():

    initRand()
    a = processArgs()

    examples = load_examples(input_dir=a.input_dir, batch_size=a.batch_size, is_training=True)
    print("examples count = %d" % examples.count)

    # inputs and targets are [batch_size, height, width, channels]
    model = create_model(examples.inputs, examples.targets)

    inputs = examples.inputs
    targets = examples.targets
    outputs = model.outputs

    with tf.name_scope("encode_images"):
        display_fetches = {
            "paths": examples.paths,
            "inputs": inputs,
            "targets": targets,
            "outputs": outputs,
        }

    tf.summary.scalar("discriminator_loss", model.discrim_loss)
    tf.summary.scalar("generator_loss_GAN", model.gen_loss_GAN)
    tf.summary.scalar("generator_loss_L1", model.gen_loss_L1)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name + "/values", var)

    for grad, var in model.discrim_grads_and_vars + model.gen_grads_and_vars:
        tf.summary.histogram(var.op.name + "/gradients", grad)

    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    server = tf.train.Server.create_local_server()
    saver = tf.train.Saver(max_to_keep=10)

    tensors_to_log = {
        "d_loss": "discriminator_loss/discrim_loss",
        "g_loss_GAN":"generator_loss/gen_loss_GAN",
        "g_loss_L1":"generator_loss/gen_loss_L1"
    }
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=progress_freq)

    max_steps = examples.steps_per_epoch * a.max_epochs
    summary_op = tf.summary.merge_all()

    hooks = [
        tf.train.StopAtStepHook(num_steps=max_steps),
        tf.train.CheckpointSaverHook(save_steps=save_freq,checkpoint_dir=a.output_dir,saver=saver),
        tf.train.SummarySaverHook(save_steps=summary_freq, summary_op=summary_op),
        logging_hook,
        tf.train.StepCounterHook(every_n_steps=progress_freq),
        SaveImageHook(output_dir=a.output_dir, fetches=display_fetches, save_steps=save_freq)
    ]

    global_step = tf.train.get_or_create_global_step()
    get_global_step = tf.train.get_global_step()

    init_op=tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    scaffold = tf.train.Scaffold(init_op)

    with tf.train.MonitoredTrainingSession(master=server.target,
                                       config=tf.ConfigProto(allow_soft_placement=True),
                                       is_chief=True,
                                       scaffold = scaffold,
                                       hooks=hooks) as sess:    
        coord = tf.train.Coordinator()

        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            while not sess.should_stop():
                out = sess.run(model)
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()
            coord.join(threads)

main()
