import tesorflow as tf
import argparse


arg = argparse()


class Graph():
    
    def __init__(self,is_training=True):
        tf.reset_default_graph()
        self.is_training = arg.is_training
        self.hidden_units = arg.hidden_units
        self.input_vocab_size = arg.input_vocab_size
        self.label_vocab_size = arg.label_vocab_size
        self.num_heads = arg.num_heads
        self.num_blocks = arg.num_blocks
        self.max_length = arg.ax_length
        self.lr= arg.lr
        self.dropout = arg.dropout_rate
        
        
        self.x = tf.placeholder(tf.int32,shape=(None,None))
        self.y = tf.placeholder(tf.int32,shape = (None,None))
        self.de_inp = tf.placeholder(tf.int32,shape=(None,None))
        
        
        # with tf.variable_scope("encoder"):
        