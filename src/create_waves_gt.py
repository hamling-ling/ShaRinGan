import numpy as np
import tensorflow as tf
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os

wave_num=10*1000
sz=256

def createGraph():
    d=tf.constant(0.23)
    b=tf.constant(0.03)
    fs=tf.constant(44100.0)
    gam=tf.constant(1.7)
    rnd_f=tf.random_uniform([1], 110, 880)[0]
    bpm=tf.constant(80.0)
    nharm=tf.floor(tf.div(fs,(2.0*rnd_f)))
    Nbeats = tf.constant(8)
    T=60.0/bpm

    nharm_i=tf.cast(nharm, dtype=tf.int32)
    An=tf.zeros([1,nharm_i], dtype=tf.float32)

    js=tf.range(1.0,nharm,1.0, dtype=tf.float32)
    tf_pi=tf.constant(np.pi, dtype=tf.float32)

    An=tf.sin(js*np.pi*d) * 2.0/(tf.pow(tf_pi,2.0) * tf.pow(js,2.0) * d * (1-d))
    dfn=tf.sqrt(1.0+js*js*b*b)

    Nt=tf.cast(fs*T, dtype=tf.int32)
    sj=tf.zeros([Nt])
    ns = tf.range(tf.cast(Nt, dtype=tf.float32), dtype=tf.float32)

    ns = tf.reshape(ns, [-1,1])
    js=tf.reshape(js, [1,-1])
    dfn=tf.reshape(dfn, [1,-1])
    sj=An*tf.exp(-gam*js*ns/fs) * tf.sin(2.0*tf_pi*ns*rnd_f*(dfn*js)/fs)
    wave=tf.reduce_sum(sj,axis=1)

    rnd_phase=tf.random_uniform([1], 0, Nt-sz-1, dtype=tf.int32)[0]

    return ns[rnd_phase:rnd_phase+sz], wave[rnd_phase:rnd_phase+sz]

def outputWave(path):
  with tf.Session() as sess:
    
    g=createGraph()
    for i in np.arange(wave_num):
        t, waves=sess.run(g)

        data = np.zeros([2,sz], dtype=np.float32)
        data[0] = waves[:sz]
        data[1] = distortion(waves[:sz])
        fn = "{0}/{1:0>4}".format(path, i) + ".bin"
        data.tofile(fn)
        print(fn, " saved")

def distortion(x):
    return np.sign(x)*(1-np.exp(-np.abs(x*8)))

out_train="../data/input/training"
out_validation = "../data/input/validation"
out_evaluation = "../data/input/evaluation"

os.makedirs("../data", exist_ok=True)
os.makedirs("../data/input", exist_ok=True)
os.makedirs(out_train, exist_ok=True)
os.makedirs(out_validation, exist_ok=True)
os.makedirs(out_evaluation, exist_ok=True)

outputWave(out_train)
outputWave(out_validation)
outputWave(out_evaluation)
