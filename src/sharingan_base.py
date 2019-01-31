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

EPS = 1e-5
SZ = 1024
TF_DTYPE = tf.float16

Examples = collections.namedtuple("Examples", "paths, inputs, targets, count, steps_per_epoch")
Model = collections.namedtuple("Model", "outputs, predict_real, predict_fake, discrim_loss, discrim_grads_and_vars, gen_loss_GAN, gen_loss_L1, gen_grads_and_vars, train")
HyperParams = collections.namedtuple("HyperParams", "lr, beta1, l1_weight, gan_weight, ngf, ndf")

def location(depth=0):
    frame = inspect.currentframe().f_back
    file = os.path.basename(frame.f_code.co_filename)
    func = frame.f_code.co_name
    line = frame.f_lineno
    return "{0}:{1}:{2}".format(file,func,line)

def batch_norm(inputs, is_training, decay = 0.9):
    eps = tf.constant(1e-5, dtype=TF_DTYPE)
    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]], dtype=TF_DTYPE))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]], dtype=TF_DTYPE))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]], dtype=TF_DTYPE), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]], dtype=TF_DTYPE), trainable=False)

    if is_training:
        batch_mean, batch_var = tf.nn.moments(inputs,[0,1,2])
        train_mean = tf.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,
                batch_mean, batch_var, beta, scale, eps)
    else:
        return tf.nn.batch_normalization(inputs,
            pop_mean, pop_var, beta, scale, eps)

def deconv_shape(input, channels_scale, out_channels=None):
  ''' twice the width, specified scale or specified number of channels '''
  if(out_channels is None):
    out_channels = int(int(input.get_shape()[-1]) * channels_scale)
  
  input_size_h = 1 #input.get_shape().as_list()[1]
  input_size_w = int(input.get_shape().as_list()[2] * 4)
  output_shape = tf.stack([tf.shape(input)[0], 
                           input_size_h,
                           input_size_w,
                           out_channels])
  return output_shape

def create_w(shape, name):
  return tf.get_variable(name, shape=shape, initializer=tf.random_normal_initializer(0, 1.0, dtype=TF_DTYPE), dtype=TF_DTYPE)

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
        raw_input = tf.cast(raw_input, TF_DTYPE)
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

def create_generator(   generator_inputs,
                        generator_outputs_channels, #not used. remove later
                        ngf,
                        is_training,
                        is_fused):
  # generator encoder filter [h, w, in ch, out ch]
  gen_w01 = create_w(shape=[1, 8,       1, ngf *  1], name="gen_w01") #1ch  -> 4ch
  gen_w02 = create_w(shape=[1, 8, ngf * 1, ngf *  2], name="gen_w02") #4ch  -> 8ch
  gen_w03 = create_w(shape=[1, 8, ngf * 2, ngf *  4], name="gen_w03") #8ch  -> 16ch
  gen_w04 = create_w(shape=[1, 8, ngf * 4, ngf *  8], name="gen_w04") #16ch -> 32ch
  gen_w05 = create_w(shape=[1, 8, ngf * 8, ngf * 16], name="gen_w05") #32ch -> 64ch
  # generator decoder filter [h, w, out ch, in ch] ... in and out are reversed!
  gen_w11 = create_w(shape=[1, 8, ngf * 1 * 8, ngf * 1 * 16], name="gen_w11") #64ch -> 32ch
  gen_w12 = create_w(shape=[1, 8, ngf * 1 * 4, ngf * 2 *  8], name="gen_w12") #64ch  -> 16ch
  gen_w13 = create_w(shape=[1, 8, ngf * 1 * 2, ngf * 2 *  4], name="gen_w13") #32ch  -> 8ch
  gen_w14 = create_w(shape=[1, 8, ngf * 1 * 1, ngf * 2 *  2], name="gen_w14") #16ch  -> 4ch
  gen_w15 = create_w(shape=[1, 8,           1, ngf * 2 *  1], name="gen_w15") #8ch   -> 1ch
  
  #encooder_1
  out = tf.nn.conv2d( generator_inputs, gen_w01, strides=(1,1,4,1), padding="SAME")
  e01 = batch_norm(out, is_training)
  #encoder_2
  out = tf.nn.leaky_relu(e01, alpha=tf.constant(0.2, dtype=TF_DTYPE))
  out = tf.nn.conv2d( out, gen_w02, strides=(1,1,4,1), padding="SAME")
  e02 = batch_norm(out, is_training)
  #encoder_3
  out = tf.nn.leaky_relu(e02, tf.constant(0.2, dtype=TF_DTYPE))
  out = tf.nn.conv2d( out, gen_w03, strides=(1,1,4,1), padding="SAME")
  e03 = batch_norm(out, is_training)
  #encoder_4
  out = tf.nn.leaky_relu(e03, tf.constant(0.2, dtype=TF_DTYPE))
  out = tf.nn.conv2d( out, gen_w04, strides=(1,1,4,1), padding="SAME")
  e04 = batch_norm(out, is_training)
  #encoder_5
  out = tf.nn.leaky_relu(e04, tf.constant(0.2, dtype=TF_DTYPE))
  out = tf.nn.conv2d( out, gen_w05, strides=(1,1,4,1), padding="SAME")
  e05 = batch_norm(out, is_training)

  print("e01", e01.get_shape()) # [b, 1, 256, 4]
  print("e02", e02.get_shape()) # [b, 1, 64,  8]
  print("e03", e03.get_shape()) # [b, 1, 16, 16]
  print("e04", e04.get_shape()) # [b, 1, 4,  32]
  print("e05", e05.get_shape()) # [b, 1, 1,  64]

  #decoder_1 [b, 1, 1, 64] => [b, 1, 4, 32]
  out = tf.nn.relu(e05)
  out_shape=deconv_shape(out, 0.5)
  print("d1 deconv2d input=", out.get_shape(), "w11=", gen_w11.get_shape())
  out = tf.nn.conv2d_transpose(out, gen_w11, strides=(1,1,4,1), output_shape=out_shape, padding="SAME")
  print("d1 deconv2d output=", out.get_shape())
  out = batch_norm(out, is_training)
  if(is_training):
    out = tf.nn.dropout(out, keep_prob=1 - 0.5)

  #decoder_2 [b, 1, 4, 32] => [b, 1, 4, 64] => [b, 1, 16, 16]
  print("d2 concatting ", out.get_shape(), " + ", e04.get_shape())
  out = tf.concat([out, e04], axis=3)
  out = tf.nn.relu(out)
  out_shape=deconv_shape(out, 0.25)
  print("d2 deconv2d input=", out.get_shape(), "w12=", gen_w12.get_shape())
  out = tf.nn.conv2d_transpose(out, gen_w12, strides=(1,1,4,1), output_shape=out_shape, padding="SAME")
  print("d2 deconv2d output=", out.get_shape())
  out = batch_norm(out, is_training)

  #decoder_3 [b, 1, 16, 16] => [b, 1, 16, 32] => [b, 1, 64, 8]
  print("d3 concatting ", out.get_shape(), " + ", e03.get_shape())
  out = tf.concat([out, e03], axis=3)
  out = tf.nn.relu(out)
  out_shape=deconv_shape(out, 0.25)
  print("d3 deconv2d input=", out.get_shape(), "w13=", gen_w13.get_shape())
  out = tf.nn.conv2d_transpose(out, gen_w13, strides=(1,1,4,1), output_shape=out_shape, padding="SAME")
  print("d3 deconv2d output=", out.get_shape())
  out = batch_norm(out, is_training)

  #decoder_4 [b, 1, 64, 8] => [b, 1, 64, 16] => [b, 1, 256, 4]
  print("d4 concatting ", out.get_shape(), " + ", e02.get_shape())
  out = tf.concat([out, e02], axis=3)
  out = tf.nn.relu(out)
  out_shape=deconv_shape(out, 0.25)
  print("d4 deconv2d input=", out.get_shape(), "w14=", gen_w14.get_shape())
  out = tf.nn.conv2d_transpose(out, gen_w14, strides=(1,1,4,1), output_shape=out_shape, padding="SAME")
  print("d4 deconv2d output=", out.get_shape())
  out = batch_norm(out, is_training)

  #decoder_5 [b, 1, 256, 4] => [b, 1, 256, 8] => [b, 1, 1024, 1]
  print("d5 concatting ", out.get_shape(), " + ", e01.get_shape())
  out = tf.concat([out, e01], axis=3)
  out = tf.nn.relu(out)
  out_shape=deconv_shape(out, 0.0, out_channels=1)
  print("d5 deconv2d input=", out.get_shape(), "w15=", gen_w15.get_shape())
  out = tf.nn.conv2d_transpose(out, gen_w15, strides=(1,1,4,1), output_shape=out_shape, padding="SAME")
  print("d5 deconv2d output=", out.get_shape())
  out = tf.tanh(out)

  return out

def create_discriminator(   discrim_inputs,
                            discrim_targets,
                            ndf,
                            is_training = True,
                            is_fused    = True):
  # discriminator params
  dis_w01 = create_w(shape=[1, 4,       2, ndf * 1], name="dis_w01")
  dis_w02 = create_w(shape=[1, 4, ndf * 1, ndf * 2], name="dis_w02")
  dis_w03 = create_w(shape=[1, 4, ndf * 2, ndf * 4], name="dis_w03")
  dis_w04 = create_w(shape=[1, 4, ndf * 4, ndf * 8], name="dis_w04")
  dis_w05 = create_w(shape=[1, 4, ndf * 8, ndf * 8], name="dis_w05")
  dis_w06 = create_w(shape=[1, 4, ndf * 8, ndf * 8], name="dis_w06")
  dis_w07 = create_w(shape=[1, 4, ndf * 8,       1], name="dis_w07")
  
  #dis0  2x [b, 1, 1024, 1] => [b, 1, 1024, 2]
  out = tf.concat([discrim_inputs, discrim_targets], axis=3)

  #dis1 [b, 1, 1024, 2] => [b, 1, 1026, 2] => [b, 1, 513, ndf]
  out = tf.pad(out, [[0, 0], [0, 0], [1, 1], [0, 0]], mode="CONSTANT")
  print("dis1 conv2d in =", out.get_shape(), ", w01=", dis_w01.get_shape())
  out = tf.nn.conv2d( out, dis_w01, strides=(1,1,2,1), padding="SAME")
  print("dis1 conv2d out=", out.get_shape())
  out = tf.nn.leaky_relu(out, tf.constant(0.2, dtype=TF_DTYPE))

  # layer_2: [b, 1, 513, ndf] => [b, 1, 515, ndf] => [b, 1, 258, ndf * 2]
  out = tf.pad(out, [[0, 0], [0, 0], [1, 1], [0, 0]], mode="CONSTANT")
  print("dis2 conv2d in =", out.get_shape(), ", w02=", dis_w02.get_shape())
  out = tf.nn.conv2d( out, dis_w02, strides=(1,1,2,1), padding="SAME")
  print("dis2 conv2d out=", out.get_shape())
  out = batch_norm(out, is_training)
  out = tf.nn.leaky_relu(out, tf.constant(0.2, dtype=TF_DTYPE))

  # layer_3: [b, 1, 258, ndf * 2] => [b, 1, 260, ndf * 2]  => [b, 1, 130, ndf * 4]
  out = tf.pad(out, [[0, 0], [0, 0], [1, 1], [0, 0]], mode="CONSTANT")
  print("dis3 conv2d in =", out.get_shape(), ", w03=", dis_w03.get_shape())
  out = tf.nn.conv2d( out, dis_w03, strides=(1,1,2,1), padding="SAME")
  print("dis3 conv2d out=", out.get_shape())
  out = batch_norm(out, is_training)
  out = tf.nn.leaky_relu(out, tf.constant(0.2, dtype=TF_DTYPE))

  # layer_4: [b, 1, 130, ndf * 2] => [b, 1, 132, ndf * 2] => [b, 1, 66, ndf * 4]
  out = tf.pad(out, [[0, 0], [0, 0], [1, 1], [0, 0]], mode="CONSTANT")
  print("dis4 conv2d in =", out.get_shape(), ", w04=", dis_w04.get_shape())
  out = tf.nn.conv2d( out, dis_w04, strides=(1,1,2,1), padding="SAME")
  print("dis4 conv2d out=", out.get_shape())
  out = batch_norm(out, is_training)
  out = tf.nn.leaky_relu(out, tf.constant(0.2, dtype=TF_DTYPE))

  # layer_5: [b, 1, 66, ndf * 2] => [b, 1, 68, ndf * 2] => [b, 1, 34, ndf * 4]
  out = tf.pad(out, [[0, 0], [0, 0], [1, 1], [0, 0]], mode="CONSTANT")
  print("dis5 conv2d in =", out.get_shape(), ", w05=", dis_w05.get_shape())
  out = tf.nn.conv2d( out, dis_w05, strides=(1,1,2,1), padding="SAME")
  print("dis5 conv2d out=", out.get_shape())
  out = batch_norm(out, is_training)
  out = tf.nn.leaky_relu(out, tf.constant(0.2, dtype=TF_DTYPE))

  # layer_6: [b, 1, 34, ndf * 4] => [b, 1, 36, ndf * 4] => [b, 1, 33, ndf * 8]
  out = tf.pad(out, [[0, 0], [0, 0], [1, 1], [0, 0]], mode="CONSTANT")
  print("dis6 conv2d in =", out.get_shape(), ", w06=", dis_w06.get_shape())
  out = tf.nn.conv2d( out, dis_w06, strides=(1,1,1,1), padding="SAME")
  print("dis6 conv2d out=", out.get_shape())
  out = batch_norm(out, is_training)
  out = tf.nn.leaky_relu(out, tf.constant(0.2, dtype=TF_DTYPE))

  # layer_7: [b, 1, 36, ndf * 8] => [b, 1, 38, ndf * 4] => [b, 1, 38, 1]
  out = tf.pad(out, [[0, 0], [0, 0], [1, 1], [0, 0]], mode="CONSTANT")
  print("dis7 conv2d in =", out.get_shape(), ", w07=", dis_w07.get_shape())
  out = tf.nn.conv2d( out, dis_w07, strides=(1,1,1,1), padding="SAME")
  print("dis7 conv2d out=", out.get_shape())
  out = batch_norm(out, is_training)
  out = tf.nn.leaky_relu(out, tf.constant(0.2, dtype=TF_DTYPE))
  #out = tf.nn.relu(out)
  out = tf.sigmoid(out)

  return out

def create_model(inputs,
                 targets,
                 hyper_params,
                 is_training = True,
                 is_fused    = True):
    with tf.variable_scope("generator"):
        out_channels = int(targets.get_shape()[-1])
        outputs = create_generator(generator_inputs           = inputs,
                                   generator_outputs_channels = out_channels,
                                   ngf                        = hyper_params.ngf,
                                   is_training                = is_training,
                                   is_fused                   = is_fused)

    # create two copies of discriminator, one for real pairs and one for fake pairs
    # they share the same underlying variables
    with tf.name_scope("real_discriminator"):
        with tf.variable_scope("discriminator"):
            # 2x [batch, height, width, channels] => [batch, 1, 30, 1]
            predict_real = create_discriminator(discrim_inputs  = inputs,
                                                discrim_targets = targets,
                                                ndf             = hyper_params.ndf,
                                                is_training     = is_training,
                                                is_fused        = is_fused)

    with tf.name_scope("fake_discriminator"):
        with tf.variable_scope("discriminator", reuse=True):
            # 2x [batch, height, width, channels] => [batch, 1, 30, 1]
            predict_fake = create_discriminator(discrim_inputs  = inputs,
                                                discrim_targets = outputs,
                                                ndf             = hyper_params.ndf,
                                                is_training     = is_training,
                                                is_fused        = is_fused)

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
        gen_loss = gen_loss_GAN * hyper_params.gan_weight + gen_loss_L1 * hyper_params.l1_weight

    with tf.name_scope("discriminator_train"):
        discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
        discrim_optim = tf.train.AdamOptimizer(hyper_params.lr, hyper_params.beta1, epsilon=EPS)
        discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
        discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

    with tf.name_scope("generator_train"):
        with tf.control_dependencies([discrim_train]):
            gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
            gen_optim = tf.train.AdamOptimizer(hyper_params.lr, hyper_params.beta1, epsilon=EPS)
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
