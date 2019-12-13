#coding:utf-8
import tensorflow as tf

img = tf.Variable(tf.random_normal([128,4,2,3]))
axis = [0,1,2]
mean,variance = tf.nn.moments(img,axis,keep_dims=True)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    mean_,variance_=sess.run([mean,variance])
    print("均值:",mean_)
    print("方差:",variance_)