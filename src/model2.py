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
    
    This is to keep TensorFlow-related components in a neat shell.
    '''

    def __init__(self, num_rels1, num_ents1, num_rels2, num_ents2, dim, batch_sizeK=1024, batch_sizeA=128, L1=False):
        self._num_relsA = num_rels1
        self._num_entsA = num_ents1
        self._num_relsB = num_rels2
        self._num_entsB = num_ents2
        self._dim = dim  # dimension of both relation and ontology. 
        self._batch_sizeK = batch_sizeK
        self._batch_sizeA = batch_sizeA
        self._epoch_loss = 0
        # margins
        self._m1 = 0.5
        self._m2 = 0.5
        self.L1 = L1
        self.build()

    @property
    def dim(self):
        return self._dim

    @property
    def batch_size(self):
        return self._batch_size    

    def build(self):
        tf.reset_default_graph()

        with tf.variable_scope("graph"):
            # Variables (matrix of embeddings/transformations)

            self._ht1 = ht1 = tf.get_variable(
                name='ht1',  # for t AND h
                shape=[self._num_entsA, self.dim],
                dtype=tf.float32)
            self._r1 = r1 = tf.get_variable(
                name='r1',
                shape=[self._num_relsA, self.dim],
                dtype=tf.float32)

            self._ht2 = ht2 = tf.get_variable(
                name='ht2',  # for t AND h
                shape=[self._num_entsB, self.dim],
                dtype=tf.float32)
            self._r2 = r2 = tf.get_variable(
                name='r2',
                shape=[self._num_relsB, self.dim],
                dtype=tf.float32)
            
            self._ht1_norm = tf.nn.l2_normalize(ht1, 1)
            self._ht2_norm = tf.nn.l2_normalize(ht2, 1)
            
            # Affine map
            self._M = M = tf.get_variable(
                name='M', 
                shape=[self.dim, self.dim],
                initializer=orthogonal_initializer(),
                dtype=tf.float32)


            self._b = bias = tf.get_variable(
                name='b', 
                shape=[self.dim],
                initializer=tf.truncated_normal_initializer,
                dtype=tf.float32)


            # Language A KM loss : [|| h + r - t ||_2 + m1 - || h + r - t ||_2]+    here [.]+ means max (. , 0)
            self._A_h_index = A_h_index = tf.placeholder(
                dtype=tf.int64,
                shape=[self._batch_sizeK],
                name='A_h_index')
            self._A_r_index = A_r_index = tf.placeholder(
                dtype=tf.int64,
                shape=[self._batch_sizeK],
                name='A_r_index')
            self._A_t_index = A_t_index = tf.placeholder(
                dtype=tf.int64,
                shape=[self._batch_sizeK],
                name='A_t_index')
            self._A_hn_index = A_hn_index = tf.placeholder(
                dtype=tf.int64,
                shape=[self._batch_sizeK],
                name='A_hn_index')
            self._A_tn_index = A_tn_index = tf.placeholder(
                dtype=tf.int64,
                shape=[self._batch_sizeK],
                name='A_tn_index')
            '''
            A_loss_matrix = tf.subtract(
                tf.add(
                    tf.batch_matmul(A_h_ent_batch, tf.reshape(A_mat_h_batch, [-1, self.dim, self.dim])),
                    A_rel_batch),
                tf.batch_matmul(A_t_ent_batch, tf.reshape(A_mat_h_batch, [-1, self.dim, self.dim]))
            )'''
            
            A_h_ent_batch = tf.nn.l2_normalize(tf.nn.embedding_lookup(ht1, A_h_index), 1)
            A_t_ent_batch = tf.nn.l2_normalize(tf.nn.embedding_lookup(ht1, A_t_index), 1)
            A_rel_batch = tf.nn.embedding_lookup(r1, A_r_index)
           
            A_hn_ent_batch = tf.nn.l2_normalize(tf.nn.embedding_lookup(ht1,A_hn_index), 1)
            A_tn_ent_batch = tf.nn.l2_normalize(tf.nn.embedding_lookup(ht1,A_tn_index), 1)
            
            # This stores h + r - t
            A_loss_matrix = tf.subtract(tf.add(A_h_ent_batch, A_rel_batch), A_t_ent_batch)
            # This stores h' + r - t' for negative samples
            A_neg_matrix = tf.subtract(tf.add(A_hn_ent_batch, A_rel_batch), A_tn_ent_batch)
            # norm
            # [||h M_hr + r - t M_tr|| + m1 - ||h' M_hr + r - t' M_tr||)]+     here [.]+ means max (. , 0)

            if self.L1:
                self._A_loss = A_loss = tf.reduce_sum(
                    tf.maximum(
                    tf.subtract(tf.add(tf.reduce_sum(tf.abs(A_loss_matrix), 1), self._m1),
                    tf.reduce_sum(tf.abs(A_neg_matrix), 1)), 
                    0.)
                ) / self._batch_sizeK
            else:
                self._A_loss = A_loss = tf.reduce_sum(
                    tf.maximum(
                    tf.subtract(tf.add(tf.sqrt(tf.reduce_sum(tf.square(A_loss_matrix), 1)), self._m1),
                    tf.sqrt(tf.reduce_sum(tf.square(A_neg_matrix), 1))), 
                    0.)
                ) / self._batch_sizeK
            
            # Language B KM loss : [|| h + r - t ||_2 + m1 - || h + r - t ||_2]+    here [.]+ means max (. , 0)
            self._B_h_index = B_h_index = tf.placeholder(
                dtype=tf.int64,
                shape=[self._batch_sizeK],
                name='B_h_index')
            self._B_r_index = B_r_index = tf.placeholder(
                dtype=tf.int64,
                shape=[self._batch_sizeK],
                name='B_r_index')
            self._B_t_index = B_t_index = tf.placeholder(
                dtype=tf.int64,
                shape=[self._batch_sizeK],
                name='B_t_index')
            self._B_hn_index = B_hn_index = tf.placeholder(
                dtype=tf.int64,
                shape=[self._batch_sizeK],
                name='B_hn_index')
            self._B_tn_index = B_tn_index = tf.placeholder(
                dtype=tf.int64,
                shape=[self._batch_sizeK],
                name='B_tn_index')
            
            B_h_ent_batch = tf.nn.l2_normalize(tf.nn.embedding_lookup(ht2, B_h_index), 1)
            B_t_ent_batch = tf.nn.l2_normalize(tf.nn.embedding_lookup(ht2, B_t_index), 1)
            B_rel_batch = tf.nn.embedding_lookup(r2, B_r_index)
           
            B_hn_ent_batch = tf.nn.l2_normalize(tf.nn.embedding_lookup(ht2,B_hn_index), 1)
            B_tn_ent_batch = tf.nn.l2_normalize(tf.nn.embedding_lookup(ht2,B_tn_index), 1)
            
            # This stores h + r - t
            B_loss_matrix = tf.subtract(tf.add(B_h_ent_batch, B_rel_batch), B_t_ent_batch)
            # This stores h' + r - t' for negative samples
            B_neg_matrix = tf.subtract(tf.add(B_hn_ent_batch, B_rel_batch), B_tn_ent_batch)
            # norm
            # [||h M_hr + r - t M_tr|| + m1 - ||h' M_hr + r - t' M_tr||)]+     here [.]+ means max (. , 0)
            
            if self.L1:
                self._B_loss = B_loss = tf.reduce_sum(
                    tf.maximum(
                    tf.subtract(tf.add(tf.reduce_sum(tf.abs(B_loss_matrix), 1), self._m1),
                    tf.reduce_sum(tf.abs(B_neg_matrix), 1)), 
                    0.)
                ) / self._batch_sizeK
            else:
                self._B_loss = B_loss = tf.reduce_sum(
                    tf.maximum(
                    tf.subtract(tf.add(tf.sqrt(tf.reduce_sum(tf.square(B_loss_matrix), 1)), self._m1),
                    tf.sqrt(tf.reduce_sum(tf.square(B_neg_matrix), 1))), 
                    0.)
                ) / self._batch_sizeK

            self._AM_index1 = AM_index1 = tf.placeholder(
                dtype=tf.int64,
                shape=[self._batch_sizeA],
                name='AM_index1')
            self._AM_index2 = AM_index2 = tf.placeholder(
                dtype=tf.int64,
                shape=[self._batch_sizeA],
                name='AM_index2')
            
            self._AM_nindex1 = AM_nindex1 = tf.placeholder(
                dtype=tf.int64,
                shape=[self._batch_sizeA],
                name='AM_nindex1')
            self._AM_nindex2 = AM_nindex2 = tf.placeholder(
                dtype=tf.int64,
                shape=[self._batch_sizeA],
                name='AM_nindex2')
            
            AM_ent1_batch = tf.nn.l2_normalize(tf.nn.embedding_lookup(ht1, AM_index1), 1)
            AM_ent2_batch = tf.nn.l2_normalize(tf.nn.embedding_lookup(ht2, AM_index2), 1)
            AM_ent1_nbatch = tf.nn.l2_normalize(tf.nn.embedding_lookup(ht1, AM_nindex1), 1)
            AM_ent2_nbatch = tf.nn.l2_normalize(tf.nn.embedding_lookup(ht2, AM_nindex2), 1)

            AM_loss_matrix = tf.subtract( tf.matmul(AM_ent1_batch, M), AM_ent2_batch)

            if self.L1:
                self._AM_loss = AM_loss = tf.reduce_sum(
                tf.reduce_sum(tf.abs(AM_loss_matrix),1)
                ) / self._batch_sizeA
            else:
                self._AM_loss = AM_loss = tf.reduce_sum(
                tf.sqrt(tf.reduce_sum(tf.square(AM_loss_matrix), 1)
                )
                ) / self._batch_sizeA

            
            # Optimizer
            self._lr = lr = tf.placeholder(tf.float32)
            self._opt = opt = tf.train.AdamOptimizer(lr)#AdagradOptimizer(lr)#GradientDescentOptimizer(lr)
            self._train_op_A = train_op_A = opt.minimize(A_loss)
            self._train_op_B = train_op_B = opt.minimize(B_loss)
            self._train_op_AM = train_op_AM = opt.minimize(AM_loss)

            # Saver
            self._saver = tf.train.Saver()