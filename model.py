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
        
        paddings = tf.ones_like(outputs)*(-2**32+1)
        outputs = tf.where(tf.equal(key_masks,0),paddings,outputs)
        
        
        if causality:
            diag_vals = tf.ones_like(outputs[0,:,:])
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()
            masks = tf.tile(tf.expand_dims(tril,0),[tf.shape(outputs)[0],1,1])
            
            paddings = tf.ones_like(masks)*(-2**32+1)
            outputs = tf.where(tf.equal(masks,0),paddings,outputs)
        
        outputs = tf.nn.softmax(outputs)
        
        
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries,axis=1)))
        query_masks = tf.tile(query_masks,[num_heads,1])
        query_masks = tf.tile(tf.expand_dims(query_masks,-1),[1,1,tf.shape(keys)[1]])
        
        
        outputs = tf.layers.dropout(outputs,rate=dropout_rate,training = tf.convert_to_tensor(is_training))
        
        outputs = tf.concat(tf.split(outputs,num_heads,axis=0),axis=2)
        
        outputs += queries
        
        outputs = normalize(outputs)
        
        return outputs



def feedforward(inputs,num_units=[2048,512],scope="multihead_attention",reuse=None):
    params = {"inputs":inputs,"filters":num_units[0],"kernel_size":1,"activation":tf.nn.relu,"use_bias":True}
    outputs = tf.layers.conv1d(**params)
    
    params = {"inputs":outputs,"filters":num_units[1],"kernel_size":1,"activation":tf.nn.relu,"use_bias":True}
    
    outputs = tf.layers.conv1d(**params)
    
    outputs += inputs
    
    outputs = normalize(outputs)
    
    return outputs

def label_smoothing(inputs,epsilon=0.1):
    K = inputs.get_shape().as_list()[-1]
    return ((1-epsilon)*inputs) + (epsilon/K)

if __name__=="__main__":
    query = tf.get_variable("query",initializer=tf.random_normal([2,3,1]))
    keys = tf.get_variable("key",initializer=tf.random_normal([2,3,1]))
    sess = tf.Session()
    outputs,init = multihead_attention(sess,query,keys,16)
    sess.run(init)
    print(sess.run(outputs))
    
   
        