#coding:utf-8
import tensorflow as tf
import copy

def normalize(inputs,epsilon = 1e-8,scope="ln",reuse=None):
    
    with tf.variable_scope(scope,reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]
        mean,variance = tf.nn.moments(inputs,[-1],keep_dims=True)
        mean,variance = tf.to_float(mean),tf.to_float(variance)
        epsilon = tf.constant(epsilon,dtype=tf.float32)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean)/((variance+epsilon)**(.5))
        outputs = gamma*normalized+beta

def embedding(inputs,vocab_size,num_units,zero_pad=True,scale=True,scope='embedding',reuse=None):
    with tf.variable_scope(scope,reuse = reuse):
        lookup_table = tf.get_variable("lookup_table",dtype=tf.float32,shape=[vocab_size,num_units],initializer=tf.contrib.layers.xavier_initializer())
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1,num_units]),lookup_table[1:,:]),0)
        
        outputs = tf.nn.embedding_lookup(lookup_table,inputs)
        
        if scale:
            outputs = outputs * (num_units**0.5)
    return outputs


def multihead_attention(sess,queries,keys,num_units=None,num_heads=8,dropout_rate=0,\
                        is_training=True,causality=False,scope="multihead_attention",reuse=None):
    with tf.variable_scope(scope,reuse=reuse):
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]
        
        Q=tf.layers.dense(queries,num_units,activation=tf.nn.relu)
        K=tf.layers.dense(keys,num_units,activation=tf.nn.relu)
        V=tf.layers.dense(keys,num_units,activation=tf.nn.relu)
        Q_ = tf.concat(tf.split(Q,num_heads,axis=2),axis=0)
        K_ = tf.concat(tf.split(K,num_heads,axis=2),axis=0)
        V_ = tf.concat(tf.split(V,num_heads,axis=2),axis=0)
        
        outputs = tf.matmul(Q_,tf.transpose(K_,[0,2,1]))
        
        outputs = outputs/(K_.get_shape().as_list()[-1]**0.5)
        
        
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys,axis=-1)))
        key_masks = tf.tile(key_masks,[num_heads,1])
        key_masks = tf.tile(tf.expand_dims(key_masks,1),[1,tf.shape(queries)])
        
        return outputs

if __name__=="__main__":
    query = tf.get_variable("query",initializer=tf.random_normal([2,3,1]))
    keys = tf.get_variable("key",initializer=tf.random_normal([2,3,1]))
    sess = tf.Session()
    outputs,init = multihead_attention(sess,query,keys,16)
    sess.run(init)
    print(sess.run(outputs))
    
   
        