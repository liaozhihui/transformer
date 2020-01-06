import tensorflow as tf

x = tf.constant(-1,shape=[5,4,3])
sha = tf.shape(x)[1]
z = tf.range(5)


y = tf.tile(tf.expand_dims(tf.range(tf.shape(x)[1]), 0), [tf.shape(x)[0], 1])



with tf.Session() as sess:
    print(sess.run(y))