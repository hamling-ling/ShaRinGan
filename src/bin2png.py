# how to run
# python bin2png.py --input_dir="./data/input/validation" --output_dir="./" --data_size 1024

import tensorflow as tf
import numpy as np
import argparse
import os
import glob
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

SZ = 1024

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", help="path to folder containing images")
parser.add_argument("--output_dir", required=True, help="where to put output files")
parser.add_argument("--data_size", type=int, default=SZ, help="specify data length 256 or 1024 typically")
a = parser.parse_args()

def save_plots(inputs, outputs, filename):

    plt.clf()
    plt.plot(inputs[0,0,:,0], linestyle='solid')
    plt.plot(outputs[0,0,:,0], linestyle='dashed')

    plt.savefig(filename)
    plt.clf()
    print(filename, " saved")

def load_examples(input_paths):

    if len(input_paths) == 0:
        raise Exception("input_dir contains no image files")

    path_queue = tf.train.string_input_producer(input_paths, shuffle=False, num_epochs=1)
    reader = tf.WholeFileReader()
    paths, contents = reader.read(path_queue)

    raw_input = tf.decode_raw(contents, tf.float32)
    raw_input = tf.reshape(raw_input,[2,1,tf.constant(a.data_size),1])

    a_images = raw_input[0]
    b_images = raw_input[1]

    return tf.train.batch([paths, a_images, b_images], batch_size=1)

input_paths = glob.glob(os.path.join(a.input_dir, "*.bin"))
input_paths.sort()

os.makedirs(a.output_dir, exist_ok=True)

path_batch, a_batch, b_batch = load_examples(input_paths)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        while not coord.should_stop():
            paths, a_images, b_images = sess.run([path_batch, a_batch, b_batch])
            fullpath = paths[0].decode("utf-8")
            dirname, filename = os.path.split(fullpath)
            fn =  os.path.splitext(filename)[0] + ".png"
            save_plots(a_images, b_images, os.path.join(a.output_dir, fn))
    except tf.errors.OutOfRangeError:
        print("an epoch finished")
    finally:
        coord.request_stop()
        coord.join(threads)

