'''Model for holding TF parts. etc.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from multiG import multiG
import pickle

# Orthogonal Initializer from
# https://github.com/OlavHN/bnlstm
def orthogonal(shape):
  flat_shape = (shape[0], np.prod(shape[1:]))
  a = np.random.normal(0.0, 1.0, flat_shape)
  u, _, v = np.linalg.svd(a, full_matrices=False)
  q = u if u.shape == flat_shape else v
  return q.reshape(shape)

def orthogonal_initializer(scale=1.0, dtype=tf.float32):
  def _initializer(shape, dtype=tf.float32, partition_info=None):
    return tf.constant(orthogonal(shape) * scale, dtype)
  return _initializer

class TFParts(object):
    '''TensorFlow-related things. 
    
    This is to keep TensorFlow-related compondescs in a neat shell.
    '''

    def __init__(self, dim, wv_dim1, wv_dim2, length1, length2, batch_sizeA=128, learning_phase=True):
        self._dim = dim  # dimension of text embedding.
        self._wv_dim1 = wv_dim1
        self._wv_dim2 = wv_dim2
        self._batch_sizeA = batch_sizeA
        self._epoch_loss = 0
        #self._desc1 = desc_embed_padded1
        #self._desc2 = desc_embed_padded2
        self._length1 = length1
        self._length2 = length2
        # margins
        self._m1 = 1.
        self._common_activation = tf.contrib.keras.activations.tanh#tf.contrib.keras.activations.linear
        self._last_activation = tf.contrib.keras.activations.tanh
        self._negative_indication_weight = -1. / batch_sizeA
        self._common_pool_size = 2
        self._att_scalar1 = length1
        self._att_scalar2 = length1 / self._common_pool_size
        self._epsilon = 0.0#1e-12
        self._learning_phase=learning_phase
        self.build()

    @property
    def dim(self):
        return self._dim

    @property
    def batch_size(self):
        return self._batch_size    
    
    def softmax_second(self, x):
        return tf.contrib.keras.activations.softmax(x, axis=-2)

    def build(self):
        tf.reset_default_graph()

        with tf.variable_scope("graph", initializer=orthogonal_initializer()):#tf.truncated_normal_initializer):
            # Variables (matrix of embeddings/transformations)
            
            self._desc1 = AM_desc1_batch = tf.placeholder(
                dtype=tf.float32,
                shape=[None, self._length1, self._wv_dim1],
                name='desc1')
            
            self._ndesc1 = AM_desc1_nbatch = tf.placeholder(
                dtype=tf.float32,
                shape=[None, self._length1, self._wv_dim1],
                name='ndesc1')
            

            self._desc2 = AM_desc2_batch = tf.placeholder(
                dtype=tf.float32,
                shape=[None, self._length2, self._wv_dim2],
                name='desc2')
            
            self._ndesc2 = AM_desc2_nbatch = tf.placeholder(
                dtype=tf.float32,
                shape=[None, self._length2, self._wv_dim2],
                name='ndesc2')

            
            
            
            self._M = M = tf.get_variable(
                name='M', 
                shape=[self.dim, self.dim],
                dtype=tf.float32)

            #gru_1 = tf.contrib.keras.layers.Bidirectional(tf.contrib.keras.layers.GRU(
            gru_1 = tf.contrib.keras.layers.GRU(
                units=self._wv_dim1,
                return_sequences=True
            )
            
            gru_3 = tf.contrib.keras.layers.GRU(
                units=self._wv_dim1,
                return_sequences=True
            )
            
            gru_5 = tf.contrib.keras.layers.GRU(
                units=self._wv_dim1,
                return_sequences=True
            )
            
            conv1 = tf.contrib.keras.layers.Conv1D(
                     filters=self._wv_dim1,
                     kernel_size=3,
                     strides=1,
                     activation=self._common_activation,
                     padding='valid',
                     use_bias=True
            )

            DS3 = tf.contrib.keras.layers.Dense(units=self._dim, activation=self._last_activation, use_bias=True)
            #DS4 = tf.contrib.keras.layers.Dense(units=self._dim, activation=self._last_activation, use_bias=True)
            
            self._att1 = att1 = tf.contrib.keras.layers.Dense(units=1, activation='tanh', use_bias=True)
            self._att2 = att2 = tf.contrib.keras.layers.Dense(units=1, activation='tanh', use_bias=True)
            self._att3 = att3 = tf.contrib.keras.layers.Dense(units=1, activation='tanh', use_bias=True)
            
            MP1 = tf.contrib.keras.layers.MaxPool1D(pool_size=self._common_pool_size,padding='valid')
            MP2 = tf.contrib.keras.layers.MaxPool1D(pool_size=self._common_pool_size,padding='valid')
            MP3 = tf.contrib.keras.layers.MaxPool1D(pool_size=self._common_pool_size,padding='valid')
            
            #tf.contrib.keras.backend.set_learning_phase(self._learning_phase)
            
            DR1 = tf.contrib.keras.layers.Dropout(0.1)
            DR2 = tf.contrib.keras.layers.Dropout(0.1)
            #DR3 = tf.contrib.keras.layers.Dropout(0.2)
            
            print('++',AM_desc1_batch.shape)
            
            #gru_+att1
            mp1_b = conv1(gru_1(AM_desc1_batch))
            mp1_nb = conv1(gru_1(AM_desc1_nbatch))
            mp2_b = conv1(gru_1(AM_desc2_batch))
            mp2_nb = conv1(gru_1(AM_desc2_nbatch))
            
            print ("++",mp1_b.shape)
            
            att1_w = tf.contrib.keras.activations.softmax(att1(mp1_b), axis=-2)
            att1_nw = tf.contrib.keras.activations.softmax(att1(mp1_nb), axis=-2)
            att2_w = tf.contrib.keras.activations.softmax(att1(mp2_b), axis=-2)
            att2_nw = tf.contrib.keras.activations.softmax(att1(mp2_nb), axis=-2)

            print ("++att",att1_w.shape)
            
            size1 = self._att_scalar1
            
            print("++size1=", size1)
            
            """
            mp1_b = MP1(tf.multiply(mp1_b, tf.scalar_mul(size1, att1_w)))
            mp1_nb = MP1(tf.multiply(mp1_nb, tf.scalar_mul(size1, att1_nw)))
            mp2_b = MP1(tf.multiply(mp2_b, tf.scalar_mul(size1, att2_w)))
            mp2_nb = MP1(tf.multiply(mp2_nb, tf.scalar_mul(size1, att2_nw)))
            """
            
            mp1_b = tf.multiply(mp1_b, tf.scalar_mul(size1, att1_w))
            mp1_nb = tf.multiply(mp1_nb, tf.scalar_mul(size1, att1_nw))
            mp2_b = tf.multiply(mp2_b, tf.scalar_mul(size1, att2_w))
            mp2_nb = tf.multiply(mp2_nb, tf.scalar_mul(size1, att2_nw))
            
            #gru_+at2
            """
            mp1_b = gru_3(mp1_b)
            mp1_nb = gru_3(mp1_nb)
            mp2_b = gru_3(mp2_b)
            mp2_nb = gru_3(mp2_nb)
            
            att1_w = tf.contrib.keras.activations.softmax(att2(mp1_b), axis=-2)
            att1_nw = tf.contrib.keras.activations.softmax(att2(mp1_nb), axis=-2)
            att2_w = tf.contrib.keras.activations.softmax(att2(mp2_b), axis=-2)
            att2_nw = tf.contrib.keras.activations.softmax(att2(mp2_nb), axis=-2)
            
            size2 = self._att_scalar2
            
            mp1_b = MP2(tf.multiply(mp1_b, tf.scalar_mul(size2, att1_w)))
            mp1_nb = MP2(tf.multiply(mp1_nb, tf.scalar_mul(size2, att1_nw)))
            mp2_b = MP2(tf.multiply(mp2_b, tf.scalar_mul(size2, att2_w)))
            mp2_nb = MP2(tf.multiply(mp2_nb, tf.scalar_mul(size2, att2_nw)))
            """
            #gru_+at3
            mp1_b = gru_5(mp1_b)
            mp1_nb = gru_5(mp1_nb)
            mp2_b = gru_5(mp2_b)
            mp2_nb = gru_5(mp2_nb)
            
            att1_w = tf.contrib.keras.activations.softmax(att3(mp1_b), axis=-2)
            att1_nw = tf.contrib.keras.activations.softmax(att3(mp1_nb), axis=-2)
            att2_w = tf.contrib.keras.activations.softmax(att3(mp2_b), axis=-2)
            att2_nw = tf.contrib.keras.activations.softmax(att3(mp2_nb), axis=-2)
            
            mp1_b = tf.multiply(mp1_b, att1_w)
            mp1_nb = tf.multiply(mp1_nb, att1_nw)
            mp2_b = tf.multiply(mp2_b, att2_w)
            mp2_nb = tf.multiply(mp2_nb, att2_nw)
            
            #last ds
            ds1_b = tf.reduce_sum(mp1_b, 1)
            ds1_nb = tf.reduce_sum(mp1_nb, 1)
            ds2_b = tf.reduce_sum(mp2_b, 1)
            ds2_nb = tf.reduce_sum(mp2_nb, 1)
            
            #print ("++",rs1_b.shape)
            
            #ds1_b = RS1(rs1_b)
            #ds1_nb = RS1(rs1_nb)
            #ds2_b = RS1(rs2_b)
            #ds2_nb = RS1(rs2_nb)

            print (self._dim)            
            print ("++",ds1_b.shape)
            
            eb_desc_batch1 = tf.nn.l2_normalize(DS3(ds1_b), dim=1)
            eb_desc_nbatch1 = tf.nn.l2_normalize(DS3(ds1_nb), dim=1)
            eb_desc_batch2 = tf.nn.l2_normalize(DS3(ds2_b), dim=1)#tf.nn.l2_normalize(DS4(ds2_b), dim=1)
            eb_desc_nbatch2 = tf.nn.l2_normalize(DS3(ds2_nb), dim=1)#tf.nn.l2_normalize(DS4(ds2_nb), dim=1)
            
            print ("++",eb_desc_batch1.shape)
            
            indicator = np.empty((self._batch_sizeA, self._batch_sizeA), dtype=np.float32)
            indicator.fill(self._negative_indication_weight)
            np.fill_diagonal(indicator, 1.)
            indicator = tf.constant(indicator)
            """
            neg_indicator = np.empty((self._batch_sizeA, self._batch_sizeA), dtype=np.float32)
            neg_indicator.fill(self._negative_indication_weight)
            neg_indicator = tf.constant(neg_indicator)
            
            self._N_loss1 = N_loss1 = tf.reduce_sum(tf.log(tf.sigmoid(tf.multiply(tf.matmul(eb_desc_batch1, tf.transpose(eb_desc_nbatch2)), neg_indicator)) + self._epsilon)) / self._batch_sizeA
            
            self._N_loss2 = N_loss2 = tf.reduce_sum(tf.log(tf.sigmoid(tf.multiply(tf.matmul(eb_desc_nbatch1, tf.transpose(eb_desc_batch2)), neg_indicator)) + self._epsilon)) / self._batch_sizeA
            
            self._N_loss = N_loss = tf.add(N_loss1, N_loss2)
            """
            self._E_loss = E_loss = tf.reduce_sum(tf.log(tf.sigmoid(tf.multiply(tf.matmul(eb_desc_batch1, tf.transpose(eb_desc_batch2)), indicator)) + self._epsilon)) / self._batch_sizeA
            

            
            # Optimizer
            self._lr = lr = tf.placeholder(tf.float32)
            self._opt = opt = tf.train.AdagradOptimizer(lr)#AdagradOptimizer(lr)#GradientDescentOptimizer(lr)
            self._train_op_E = train_op_E = opt.minimize( -E_loss)
            
            # Text embedder
            self._dump_embed1 = eb_desc_batch1
            self._dump_embed2 = eb_desc_batch2

            # Saver
            self._saver = tf.train.Saver()