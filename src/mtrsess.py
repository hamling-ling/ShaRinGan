import tensorflow as tf
import numpy as np
import glob
import os

sz=256 # signel data length
epoch = 2
max_steps = 3 #exit loop at this step num even if epoch num not reached

def createGraph():
    in_dir="../data/input/evaluation"
    input_paths = glob.glob(os.path.join(in_dir, "*.bin"))
    input_paths.sort()
    print("load ", len(input_paths), "files")
    queue = tf.train.string_input_producer(input_paths, num_epochs=epoch)
    reader = tf.WholeFileReader()
    key, value = reader.read(queue)
    image = tf.decode_raw(value, tf.float32)
    image = tf.reshape(image,[2,1,tf.constant(sz),1])

    image = tf.multiply(image, tf.constant(0.5))
    batch = tf.train.batch([image], batch_size=1)
    return batch

tf.set_random_seed(seed=0)

# define graph
model = createGraph()

# create a one process cluster with an in-process server
server = tf.train.Server.create_local_server()
hooks = [tf.train.StopAtStepHook(num_steps=max_steps)]
global_step = tf.train.get_or_create_global_step()
get_global_step = tf.train.get_global_step()
increment_global_step = tf.assign(global_step, global_step+1)

init_op=tf.group(tf.global_variables_initializer(),
                 tf.local_variables_initializer())
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
            print("before global step=", tf.train.global_step(sess, tf.train.get_global_step()))
            out, gs = sess.run([model, increment_global_step])
            print("after global step=", gs)
            #if(gs == 2):
            #    break
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()
        coord.join(threads)

    img = out[0] # take one from batch

    print(out.shape)
