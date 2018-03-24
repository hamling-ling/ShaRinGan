from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os
import inspect
import json
import glob
import random
import collections
import math
import time
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

kSz = 256

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", help="path to folder containing images")
parser.add_argument("--output_dir", required=True, help="where to put output files")
parser.add_argument("--checkpoint", default=None, help="directory with checkpoint to resume training from or use for testing")

parser.add_argument("--max_steps", type=int, help="number of training steps (0 to disable)")
parser.add_argument("--max_epochs", type=int, help="number of training epochs")

parser.add_argument("--batch_size", type=int, default=1, help="number of images in batch")
parser.add_argument("--ngf", type=int, default=64, help="number of generator filters in first conv layer")
parser.add_argument("--ndf", type=int, default=64, help="number of discriminator filters in first conv layer")

a = parser.parse_args()

EPS = 1e-12
SZ = 256

save_freq=5000
display_freq=0
progress_freq=50
lr=0.0002
beta1=0.5
l1_weight=100.0
gan_weight=1.0

Examples = collections.namedtuple("Examples", "paths, inputs, targets, count, steps_per_epoch")
Model = collections.namedtuple("Model", "outputs, predict_real, predict_fake, discrim_loss, discrim_grads_and_vars, gen_loss_GAN, gen_loss_L1, gen_grads_and_vars, train")

def location(depth=0):
    frame = inspect.currentframe().f_back
    file = os.path.basename(frame.f_code.co_filename)
    func = frame.f_code.co_name
    line = frame.f_lineno
    return "{0}:{1}:{2}".format(file,func,line)

def discrim_conv(batch_input, out_channels, stride):
    ### [[0,0],[0,0],[1,1],[0,0]] => [b, h, w+2, c]
    padded_input = tf.pad(batch_input, [[0, 0], [0, 0], [1, 1], [0, 0]], mode="CONSTANT")
    ### (0,stride)
    return tf.layers.conv2d(padded_input, out_channels, kernel_size=[1,4], strides=(stride, stride), padding="valid", kernel_initializer=tf.random_normal_initializer(0, 0.02))


def gen_conv(batch_input, out_channels):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    ### strides=(0,2)
    return tf.layers.conv2d(batch_input, out_channels, kernel_size=[1,4], strides=(1, 2), padding="same", kernel_initializer=initializer)


def gen_deconv(batch_input, out_channels):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    ### strides to (0,2)
    return tf.layers.conv2d_transpose(batch_input, out_channels, kernel_size=[1,4], strides=(1, 2), padding="same", kernel_initializer=initializer)


def lrelu(x, a):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


def batchnorm(inputs):
    return tf.layers.batch_normalization(inputs, axis=3, epsilon=1e-5, momentum=0.1, training=True, gamma_initializer=tf.random_normal_initializer(1.0, 0.02))


def load_examples():
    if a.input_dir is None or not os.path.exists(a.input_dir):
        raise Exception("input_dir does not exist")

    input_paths = glob.glob(os.path.join(a.input_dir, "*.bin"))
    input_paths.sort()

    if len(input_paths) == 0:
        raise Exception("input_dir contains no image files")

    with tf.name_scope("load_images"):
        path_queue = tf.train.string_input_producer(input_paths, shuffle=True)#shuffle only for training
        reader = tf.WholeFileReader()
        paths, contents = reader.read(path_queue)

        raw_input = tf.decode_raw(contents, tf.float32)
        raw_input = tf.reshape(raw_input,[2,1,tf.constant(SZ),1])

        # break apart image pair and move to range [-1, 1]
        width = tf.shape(raw_input)[1] # [height, width, channels]
        a_images = raw_input[0]
        b_images = raw_input[1]

        inputs, targets = [a_images, b_images]

    with tf.name_scope("input_images"):
        input_images = inputs

    with tf.name_scope("target_images"):
        target_images = targets

    paths_batch, inputs_batch, targets_batch = tf.train.batch([paths, input_images, target_images], batch_size=a.batch_size)
    steps_per_epoch = int(math.ceil(len(input_paths) / a.batch_size))

    return Examples(
        paths=paths_batch,
        inputs=inputs_batch,
        targets=targets_batch,
        count=len(input_paths),
        steps_per_epoch=steps_per_epoch,
    )


def create_generator(generator_inputs, generator_outputs_channels):
    layers = []

    # encoder_1: [batch, 1, 256, in_channels] => [batch, 1, 128, ngf]
    with tf.variable_scope("encoder_1"):
        output = gen_conv(generator_inputs, a.ngf)
        layers.append(output)

    layer_specs = [
        a.ngf * 2, # encoder_2: [batch, 1, 128, ngf] => [batch, 1, 64, ngf * 2]
        a.ngf * 4, # encoder_3: [batch, 1, 64, ngf * 2] => [batch, 1, 32, ngf * 4]
        a.ngf * 8, # encoder_4: [batch, 1, 32, ngf * 4] => [batch, 1, 16, ngf * 8]
        a.ngf * 8, # encoder_5: [batch, 1, 16, ngf * 8] => [batch, 1, 8, ngf * 8]
        a.ngf * 8, # encoder_6: [batch, 1, 8, ngf * 8] => [batch, 1, 4, ngf * 8]
        a.ngf * 8, # encoder_7: [batch, 1, 4, ngf * 8] => [batch, 1, 2, ngf * 8]
        a.ngf * 8, # encoder_8: [batch, 1, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
    ]

    for out_channels in layer_specs:
        with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
            rectified = lrelu(layers[-1], 0.2)
            # [batch, in_height, in_width, in_channels] => [batch, in_height, in_width/2, out_channels]
            convolved = gen_conv(rectified, out_channels)
            output = batchnorm(convolved)
            layers.append(output)

    layer_specs = [
        (a.ngf * 8, 0.5),   # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 1, 2, ngf * 8 * 2]
        (a.ngf * 8, 0.5),   # decoder_7: [batch, 1 2, ngf * 8 * 2] => [batch, 1, 4, ngf * 8 * 2]
        (a.ngf * 8, 0.5),   # decoder_6: [batch, 1, 4, ngf * 8 * 2] => [batch, 1, 8, ngf * 8 * 2]
        (a.ngf * 8, 0.0),   # decoder_5: [batch, 1, 8, ngf * 8 * 2] => [batch, 1, 16, ngf * 8 * 2]
        (a.ngf * 4, 0.0),   # decoder_4: [batch, 1, 16, ngf * 8 * 2] => [batch, 1, 32, ngf * 4 * 2]
        (a.ngf * 2, 0.0),   # decoder_3: [batch, 1, 32, ngf * 4 * 2] => [batch, 1, 64, ngf * 2 * 2]
        (a.ngf, 0.0),       # decoder_2: [batch, 1, 64, ngf * 2 * 2] => [batch, 1, 128, ngf * 2]
    ]

    num_encoder_layers = len(layers)
    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1
        with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
            if decoder_layer == 0:
                # first decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer
                input = layers[-1]
            else:
                input = tf.concat([layers[-1], layers[skip_layer]], axis=3)

            rectified = tf.nn.relu(input)
            # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
            output = gen_deconv(rectified, out_channels)
            output = batchnorm(output)

            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)

            layers.append(output)

    # decoder_1: [batch, 1, 128, ngf * 2] => [batch, 1, 256, generator_outputs_channels]
    with tf.variable_scope("decoder_1"):
        input = tf.concat([layers[-1], layers[0]], axis=3)
        rectified = tf.nn.relu(input)
        output = gen_deconv(rectified, generator_outputs_channels)
        output = tf.tanh(output)
        layers.append(output)

    return layers[-1]


def create_model(inputs, targets):
    def create_discriminator(discrim_inputs, discrim_targets):
        n_layers = 3
        layers = []

        # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
        input = tf.concat([discrim_inputs, discrim_targets], axis=3)

        # layer_1: [batch, 1, 256, in_channels * 2] => [batch, 1, 128, ndf]
        with tf.variable_scope("layer_1"):
            convolved = discrim_conv(input, a.ndf, stride=2)
            rectified = lrelu(convolved, 0.2)
            layers.append(rectified)

        # layer_2: [batch, 1, 128, ndf] => [batch, 1, 64, ndf * 2]
        # layer_3: [batch, 1, 64, ndf * 2] => [batch, 1, 32, ndf * 4]
        # layer_4: [batch, 1, 32, ndf * 4] => [batch, 1, 31, ndf * 8]
        for i in range(n_layers):
            with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                out_channels = a.ndf * min(2**(i+1), 8)
                stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
                convolved = discrim_conv(layers[-1], out_channels, stride=stride)
                normalized = batchnorm(convolved)
                rectified = lrelu(normalized, 0.2)
                layers.append(rectified)

        # layer_5: [batch, 1, 31, ndf * 8] => [batch, 1, 30, 1]
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            convolved = discrim_conv(rectified, out_channels=1, stride=1)
            output = tf.sigmoid(convolved)
            layers.append(output)

        return layers[-1]

    with tf.variable_scope("generator"):
        out_channels = int(targets.get_shape()[-1])
        outputs = create_generator(inputs, out_channels)

    # create two copies of discriminator, one for real pairs and one for fake pairs
    # they share the same underlying variables
    with tf.name_scope("real_discriminator"):
        with tf.variable_scope("discriminator"):
            # 2x [batch, height, width, channels] => [batch, 1, 30, 1]
            predict_real = create_discriminator(inputs, targets)

    with tf.name_scope("fake_discriminator"):
        with tf.variable_scope("discriminator", reuse=True):
            # 2x [batch, height, width, channels] => [batch, 1, 30, 1]
            predict_fake = create_discriminator(inputs, outputs)

    with tf.name_scope("discriminator_loss"):
        # minimizing -tf.log will try to get inputs to 1
        # predict_real => 1
        # predict_fake => 0
        discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))

    with tf.name_scope("generator_loss"):
        # predict_fake => 1
        # abs(targets - outputs) => 0
        gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))
        gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs))
        gen_loss = gen_loss_GAN * gan_weight + gen_loss_L1 * l1_weight

    with tf.name_scope("discriminator_train"):
        discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
        discrim_optim = tf.train.AdamOptimizer(lr, beta1)
        discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
        discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

    with tf.name_scope("generator_train"):
        with tf.control_dependencies([discrim_train]):
            gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
            gen_optim = tf.train.AdamOptimizer(lr, beta1)
            gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
            gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_L1])

    global_step = tf.train.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step+1)

    return Model(
        predict_real=predict_real,
        predict_fake=predict_fake,
        discrim_loss=ema.average(discrim_loss),
        discrim_grads_and_vars=discrim_grads_and_vars,
        gen_loss_GAN=ema.average(gen_loss_GAN),
        gen_loss_L1=ema.average(gen_loss_L1),
        gen_grads_and_vars=gen_grads_and_vars,
        outputs=outputs,
        train=tf.group(update_losses, incr_global_step, gen_train),
    )

def save_plots(inputs, outputs, targets, filename):

    plt.clf()
    if(64 <= len(inputs)):
        for i in range(min(64, len(inputs))):
            plt.subplot(8,8,i+1)
            plt.plot(inputs[i,0,:,0], linestyle='solid')
            plt.plot(outputs[i,0,:,0], linestyle='dashed')
            plt.plot(targets[i,0,:,0], linestyle='dot')
    else:
        plt.plot(inputs[0,0,:,0], linestyle='solid')
        plt.plot(outputs[0,0,:,0], linestyle='dashed')
        plt.plot(targets[0,0,:,0], linestyle='dotted')

    plt.savefig(filename)
    plt.clf()
    print(filename, " saved")

def save_images(fetches, step=None):
    image_dir = os.path.join(a.output_dir, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    if(step is None):
        step = 0

    fn = os.path.join(image_dir, "step{0:0>4}.png".format(step))
    save_plots(fetches["inputs"], fetches["outputs"], fetches["targets"], fn)

def main():
    seed = 0
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)

    for k, v in a._get_kwargs():
        print(k, "=", v)

    with open(os.path.join(a.output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(a), sort_keys=True, indent=4))

    examples = load_examples()
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

    saver = tf.train.Saver(max_to_keep=1)

    sv = tf.train.Supervisor(logdir=None, save_summaries_secs=0, saver=None)
    with sv.managed_session() as sess:
        print("parameter_count =", sess.run(parameter_count))

        if a.checkpoint is not None:
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            saver.restore(sess, checkpoint)

        max_steps = 2**32
        if a.max_epochs is not None:
            max_steps = examples.steps_per_epoch * a.max_epochs
        if a.max_steps is not None:
            max_steps = a.max_steps

        # training
        start = time.time()

        for step in range(max_steps):
            def should(freq):
                return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)

            fetches = {
                "train": model.train,
                "global_step": sv.global_step,
            }

            if should(progress_freq):
                fetches["discrim_loss"] = model.discrim_loss
                fetches["gen_loss_GAN"] = model.gen_loss_GAN
                fetches["gen_loss_L1"] = model.gen_loss_L1

            if should(display_freq):
                fetches["display"] = display_fetches

            results = sess.run(fetches)

            if should(display_freq):
                print("saving display images")
                save_images(results["display"], step=results["global_step"])

            if should(progress_freq):
                # global_step will have the correct step count if we resume from a checkpoint
                train_epoch = math.ceil(results["global_step"] / examples.steps_per_epoch)
                train_step = (results["global_step"] - 1) % examples.steps_per_epoch + 1
                rate = (step + 1) * a.batch_size / (time.time() - start)
                remaining = (max_steps - step) * a.batch_size / rate
                print("progress  epoch %d  step %d  image/sec %0.1f  remaining %dm" % (train_epoch, train_step, rate, remaining / 60))
                print("discrim_loss", results["discrim_loss"])
                print("gen_loss_GAN", results["gen_loss_GAN"])
                print("gen_loss_L1", results["gen_loss_L1"])

            if should(save_freq):
                print("saving model")
                saver.save(sess, os.path.join(a.output_dir, "model"), global_step=sv.global_step)

            if sv.should_stop():
                break


main()
