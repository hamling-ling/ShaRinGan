import tensorflow as tf
import numpy as np
import os

def create_generator(generator_inputs):
    # encoder_1: [batch, 1, 1024, 1] => [batch, 1, 512, 64]
    with tf.variable_scope("encoder_1"):
        initializer = tf.random_normal_initializer(0, 0.02)
        output = tf.layers.conv2d(generator_inputs, 64, kernel_size=[1,4], strides=(1, 2), padding="same", kernel_initializer=initializer)
    with tf.variable_scope("encoder"):
        rectified = tf.nn.leaky_relu(output)
        # [1, 1, 512, 64] => [1, 1, 256, 128]
        initializer = tf.random_normal_initializer(0, 0.02)
        convolved = tf.layers.conv2d(rectified, 128, kernel_size=[1,4], strides=(1, 2), padding="same", kernel_initializer=initializer)
        output = tf.layers.batch_normalization(convolved, axis=3, epsilon=1e-5, momentum=0.1, training=False, fused=False, gamma_initializer=tf.random_normal_initializer(1.0, 0.02))
        return output

def create_generator2(generator_inputs):
    with tf.variable_scope("encoder"):
        initializer = tf.random_normal_initializer(0, 0.02)
        convolved = tf.layers.conv2d(generator_inputs, 64, kernel_size=[1,4], strides=(1, 2), padding="same", kernel_initializer=initializer)
        output = tf.layers.batch_normalization(convolved, axis=3, epsilon=1e-5, momentum=0.1, training=False, fused=False, gamma_initializer=tf.random_normal_initializer(1.0, 0.02))
        return output


def main():

    tf.reset_default_graph()

    in_op = tf.placeholder("float", [1, 1, 1024, 1], name="input")
    graph = create_generator2(in_op)
    
    saver = tf.train.Saver()
    with tf.Session() as sess:
        inputs_feed = tf.ones([1, 1, 1024, 1])
        sess.run(tf.global_variables_initializer())
        value = sess.run(graph, feed_dict={in_op:inputs_feed.eval()})
        saver.save(sess, "./test.ckpt")

        graph_name = graph.name.split(":")[0] # remove last ":0"
        print("graph.name=", graph_name)
        cmdstr = "please execute following command\n" + "mvNCCompile test.ckpt.meta -in=input -on {0} -s 12".format(graph_name)
        print(cmdstr)

main()
