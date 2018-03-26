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
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from abc import ABCMeta, abstractmethod

tf.logging.set_verbosity(tf.logging.INFO)

EPS = 1e-12
SZ = 256

save_freq=1000
summary_freq=1000
display_freq=0
progress_freq=50
lr=0.0002
beta1=0.5
l1_weight=100.0
gan_weight=1.0
ngf=64
ndf=64

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


def load_examples(input_dir, batch_size, is_training=True):
    if input_dir is None or not os.path.exists(input_dir):
        raise Exception("input_dir does not exist")

    input_paths = glob.glob(os.path.join(input_dir, "*.bin"))
    input_paths.sort()

    if len(input_paths) == 0:
        raise Exception("input_dir contains no image files ", input_dir)

    with tf.name_scope("load_images"):
        if(is_training):
            #shuffle only for training
            path_queue = tf.train.string_input_producer(input_paths, shuffle=True)
        else:
            path_queue = tf.train.string_input_producer(input_paths, shuffle=False, num_epochs=1)
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

    paths_batch, inputs_batch, targets_batch = tf.train.batch([paths, input_images, target_images], batch_size=batch_size)
    steps_per_epoch = int(math.ceil(len(input_paths) / batch_size))

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
        output = gen_conv(generator_inputs, ngf)
        layers.append(output)

    layer_specs = [
        ngf * 2, # encoder_2: [batch, 1, 128, ngf] => [batch, 1, 64, ngf * 2]
        ngf * 4, # encoder_3: [batch, 1, 64, ngf * 2] => [batch, 1, 32, ngf * 4]
        ngf * 8, # encoder_4: [batch, 1, 32, ngf * 4] => [batch, 1, 16, ngf * 8]
        ngf * 8, # encoder_5: [batch, 1, 16, ngf * 8] => [batch, 1, 8, ngf * 8]
        ngf * 8, # encoder_6: [batch, 1, 8, ngf * 8] => [batch, 1, 4, ngf * 8]
        ngf * 8, # encoder_7: [batch, 1, 4, ngf * 8] => [batch, 1, 2, ngf * 8]
        ngf * 8, # encoder_8: [batch, 1, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
    ]

    for out_channels in layer_specs:
        with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
            rectified = lrelu(layers[-1], 0.2)
            # [batch, in_height, in_width, in_channels] => [batch, in_height, in_width/2, out_channels]
            convolved = gen_conv(rectified, out_channels)
            output = batchnorm(convolved)
            layers.append(output)

    layer_specs = [
        (ngf * 8, 0.5),   # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 1, 2, ngf * 8 * 2]
        (ngf * 8, 0.5),   # decoder_7: [batch, 1 2, ngf * 8 * 2] => [batch, 1, 4, ngf * 8 * 2]
        (ngf * 8, 0.5),   # decoder_6: [batch, 1, 4, ngf * 8 * 2] => [batch, 1, 8, ngf * 8 * 2]
        (ngf * 8, 0.0),   # decoder_5: [batch, 1, 8, ngf * 8 * 2] => [batch, 1, 16, ngf * 8 * 2]
        (ngf * 4, 0.0),   # decoder_4: [batch, 1, 16, ngf * 8 * 2] => [batch, 1, 32, ngf * 4 * 2]
        (ngf * 2, 0.0),   # decoder_3: [batch, 1, 32, ngf * 4 * 2] => [batch, 1, 64, ngf * 2 * 2]
        (ngf, 0.0),       # decoder_2: [batch, 1, 64, ngf * 2 * 2] => [batch, 1, 128, ngf * 2]
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
            convolved = discrim_conv(input, ndf, stride=2)
            rectified = lrelu(convolved, 0.2)
            layers.append(rectified)

        # layer_2: [batch, 1, 128, ndf] => [batch, 1, 64, ndf * 2]
        # layer_3: [batch, 1, 64, ndf * 2] => [batch, 1, 32, ndf * 4]
        # layer_4: [batch, 1, 32, ndf * 4] => [batch, 1, 31, ndf * 8]
        for i in range(n_layers):
            with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                out_channels = ndf * min(2**(i+1), 8)
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
        discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)), name="discrim_loss")

    with tf.name_scope("generator_loss"):
        # predict_fake => 1
        # abs(targets - outputs) => 0
        gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS), name="gen_loss_GAN")
        gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs), name="gen_loss_L1")
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
            plt.plot(targets[i,0,:,0], linestyle='dotted')
    else:
        plt.plot(inputs[0,0,:,0], linestyle='solid')
        plt.plot(outputs[0,0,:,0], linestyle='dashed')
        plt.plot(targets[0,0,:,0], linestyle='dotted')

    plt.savefig(filename)
    plt.clf()
    print(filename, " saved")

def save_images(output_dir, fetches, step=None):
    image_dir = os.path.join(output_dir, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    if(step is None):
        step = 0

    fn = os.path.join(image_dir, "step{0:0>4}.png".format(step))
    save_plots(fetches["inputs"], fetches["outputs"], fetches["targets"], fn)

class StepCountHook(tf.train.SessionRunHook):

    __metaclass__ = ABCMeta

    def __init__(self, op, hook_steps):
        self._hook_steps = hook_steps
        self._op = op

    def begin(self):
        self._step = 0

    def before_run(self, run_context):
        if(self._step % self._hook_steps == 0):
            if(self._op is not None):
                return tf.train.SessionRunArgs(self._op)

    def after_run(self, run_context, run_values):
        if(run_values is not None and run_values.results is not None):
            val = run_values.results
            self.ellapsed(val=val, step=self._step)
        self._step += 1

    @abstractmethod
    def ellapsed(self, val, step):
        return

class SaveImageHook(StepCountHook):
    def __init__(self, output_dir, fetches, save_steps):
        self._output_dir = output_dir
        super(SaveImageHook, self).__init__(op=fetches, hook_steps=save_steps)

    def ellapsed(self, val, step):
        save_images(output_dir=self._output_dir, fetches=val, step=step)

class ProgressLoggingHook(StepCountHook):
    def __init__(self, log_steps, max_steps):
        self._max_steps = max_steps
        op = tf.constant(3)# have to set something. work around
        super(ProgressLoggingHook, self).__init__(op=op, hook_steps=log_steps)

    def ellapsed(self, val, step):
        percent=100.0*step/self._max_steps
        print("progress={0}% ({1}/{2})".format(percent, step, self._max_steps))
