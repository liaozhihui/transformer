#coding:utf-8
import tensorflow as tf

#img = tf.Variable(tf.random_normal([128,4,2,3]))
#axis = [0,1,2]
#mean,variance = tf.nn.moments(img,axis,keep_dims=True)
#init = tf.global_variables_initializer()
#with tf.Session() as sess:
    #sess.run(init)
    #mean_,variance_=sess.run([mean,variance])
    #print("均值:",mean_)
    #print("方差:",variance_)

t = tf.Variable(tf.random_normal([2,3,5]))
print(t.get_shape())
t2=tf.expand_dims(t,1)
print(t2.get_shape())
#with tf.Session() as sess:
    #sess.run(tf.global_variables_initializer())
    #print(sess.run(t))