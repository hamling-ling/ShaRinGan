import tensorflow as tf
import numpy as np
import os

def create_generator(input):
    with tf.variable_scope("encoder"):
        ini = tf.random_normal_initializer(0, 0.02)
        out = tf.layers.conv2d(input, 2, kernel_size=[1,4], strides=(1, 2), padding="same", kernel_initializer=ini)
        out = tf.layers.conv2d(out, 4, kernel_size=[1,4], strides=(1, 2), padding="same", kernel_initializer=ini)
    return out

def save_last_node_name(node_name):
    f = open('last_node_name.txt', 'w')
    f.write(node_name)
    f.close()
    print(node_name, " saved in last_node_name.txt")

def main():

    tf.reset_default_graph()

    in_op = tf.placeholder("float", [1, 1, 1024, 1], name="input")

    graph = create_generator(in_op)

    save_last_node_name(graph.name.split(":")[0])
    
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
