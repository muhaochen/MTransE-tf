''' Module for training TF parts.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import time
from numpy import linalg as LA
import pdb

from multiG import multiG 
import gru_encoder


class Trainer(object):
    def __init__(self):
        self.batch_sizeA=32
        self.dim=64
        self._m1 = 0.5
        self.multiG = None
        self.tf_parts = None
        self.save_path = 'this-gru.ckpt'
        self.multiG_save_path = 'this-multiG2.bin'
        self.L1=False
        self.sess = None
        self.debug = True
        self.valid = True
        self.valid_loss = []
        self.valid_neg_loss = []
        self.lr_decay_coef = 1.
        self.lr_decay_stop = 0.0625
        self.lr_decay_thres = 0.001
        self.lr_decay = True
        self.lr = 1.
        self.pos_max_loss = -float('inf')
        self.neg_min_loss = float('inf')

    def build(self, multiG, dim=64, batch_sizeA=128, m1=1., save_path = 'this-gru.ckpt', multiG_save_path = 'this-multiG2.bin', L1=False):
        self.multiG = multiG
        self.dim = self.multiG.desc_dim = dim
        #self.multiG.KG1.wv_dim = self.multiG.KG2.wv_dim = wv_dim
        self.batch_sizeA = self.multiG.batch_sizeA = batch_sizeA
        self.multiG_save_path = multiG_save_path
        self.save_path = save_path
        self.L1 = self.multiG.L1 = L1
        self.tf_parts = gru_encoder.TFParts(
                          dim=dim,
                          wv_dim1=self.multiG.KG1.wv_dim,
                          wv_dim2=self.multiG.KG2.wv_dim,
                          length1=self.multiG.KG1.desc_length,
                          length2=self.multiG.KG2.desc_length,
                          batch_sizeA=batch_sizeA,
                          learning_phase=True)
        self.tf_parts._m1 = m1
        self.sess = sess = tf.Session()
        sess.run(tf.initialize_all_variables())

    def gen_AM_batch(self, forever=False, shuffle=True):
        multiG = self.multiG
        l = multiG.num_align_desc()
        while True:
            align = multiG.align_desc
            if shuffle:
                np.random.shuffle(align)
            for i in range(0, l, self.batch_sizeA):
                batch = align[i: i+self.batch_sizeA, :]
                if batch.shape[0] < self.batch_sizeA:
                    batch = np.concatenate((batch, align[:self.batch_sizeA - batch.shape[0]]), axis=0)
                    assert batch.shape[0] == self.batch_sizeA
                nbatch = multiG.sample_false_pair(self.batch_sizeA)
                e1_batch, e2_batch, e1_nbatch, e2_nbatch = batch[:, 0], batch[:, 1], nbatch[:, 0], nbatch[:, 1]
                yield e1_batch.astype(np.int64), e2_batch.astype(np.int64), e1_nbatch.astype(np.int64), e2_nbatch.astype(np.int64)
            if not forever:
                break
    
    def gen_AM_batch_no_neg(self, forever=False, shuffle=True):
        multiG = self.multiG
        l = multiG.num_align_desc()
        while True:
            align = multiG.align_desc
            if shuffle:
                np.random.shuffle(align)
            for i in range(0, l, self.batch_sizeA):
                batch = align[i: i+self.batch_sizeA, :]
                if batch.shape[0] < self.batch_sizeA:
                    batch = np.concatenate((batch, align[:self.batch_sizeA - batch.shape[0]]), axis=0)
                    assert batch.shape[0] == self.batch_sizeA
                #nbatch = multiG.sample_false_pair(self.batch_sizeA)
                e1_batch, e2_batch = batch[:, 0], batch[:, 1]
                yield e1_batch.astype(np.int64), e2_batch.astype(np.int64)
            if not forever:
                break

    def train1epoch_AM(self, sess, num_AM_batch, lr, epoch):
        #this_gen_AM_batch = self.gen_AM_batch(forever=True)
        this_gen_AM_batch = self.gen_AM_batch_no_neg(forever=True)
        
        this_loss = []
        
        loss_E = loss_CE = 0

        for batch_id in range(num_AM_batch):
            # Optimize loss A
            #e1_index, e2_index, e1_nindex, e2_nindex  = next(this_gen_AM_batch)
            e1_index, e2_index  = next(this_gen_AM_batch)
            _, loss_E = sess.run([self.tf_parts._train_op_E, self.tf_parts._E_loss],#, self.tf_parts._N_loss],
                    feed_dict={self.tf_parts._desc1: self.multiG.KG1.desc_embed_padded[e1_index], 
                               self.tf_parts._desc2: self.multiG.KG2.desc_embed_padded[e2_index],
                               #self.tf_parts._ndesc1: self.multiG.KG1.desc_embed_padded[e1_nindex], 
                               #self.tf_parts._ndesc2: self.multiG.KG2.desc_embed_padded[e2_nindex],
                               self.tf_parts._lr: lr * self.lr_decay_coef})
            # Observe total loss
            batch_loss = [loss_E]
            if len(this_loss) == 0:
                this_loss = np.array(batch_loss)
            else:
                this_loss += np.array(batch_loss)
            if ((batch_id + 1) % 50 == 0) or batch_id == num_AM_batch - 1:
                print('\rprocess: %d / %d. Epoch %d' % (batch_id+1, num_AM_batch+1, epoch))
        this_total_loss = np.sum(this_loss)
        print("AM Loss of epoch", epoch, ":", this_total_loss)
        print([l for l in this_loss])
        if self.debug and (epoch < 50 or epoch % 2 == 1):
            print('\n=========================\nDebug:\n')
            ht1, ht2 = sess.run([self.tf_parts._dump_embed1, self.tf_parts._dump_embed2],
                            feed_dict={self.tf_parts._desc1: self.multiG.KG1.desc_embed_padded[e1_index],
                                       self.tf_parts._desc2: self.multiG.KG2.desc_embed_padded[e2_index]})
            gap = np.random.randint(len(ht1) - 2) + 1
            debug_avg_loss = 0. 
            #for x in (np.dot(ht1,M)-ht2):
            for x in (np.multiply(ht1,ht2)):
                debug_avg_loss += np.sum(x)
            debug_avg_loss /= len(e1_index)
            print('debug avg loss=', debug_avg_loss)
            debug_avg_loss = 0. 
            #for x in (np.dot(ht1,M)-ht2):
            for x in np.multiply(ht1, np.concatenate((ht2[gap:], ht2[:gap]))):
                debug_avg_loss += np.sum(x)
            debug_avg_loss /= len(e1_index)
            print('debug neg loss=', debug_avg_loss)
            print('\n=========================\n')
        if self.valid:
            ht1, ht2 = sess.run([self.tf_parts._dump_embed1, self.tf_parts._dump_embed2],
                            feed_dict={self.tf_parts._desc1: self.multiG.KG1.desc_embed_padded[self.multiG.align_valid[:,0]],
                                       self.tf_parts._desc2: self.multiG.KG2.desc_embed_padded[self.multiG.align_valid[:,1]]})
            valid_avg_loss = 0.
            for x in (np.multiply(ht1,ht2)):
                valid_avg_loss += np.sum(x)
            valid_avg_loss /= len(ht1)
            print('valid avg loss=', valid_avg_loss)
            if self.lr_decay:
                self.valid_loss.append(valid_avg_loss)
                if valid_avg_loss > self.pos_max_loss:
                    self.pos_max_loss = valid_avg_loss
            valid_avg_loss = 0.
            #for x in (np.dot(ht1,M)-ht2):
            gap = np.random.randint(len(ht1) - 2) + 1
            for x in np.multiply(ht1, np.concatenate((ht2[gap:], ht2[:gap]))):
                valid_avg_loss += np.sum(x)
            valid_avg_loss /= len(ht1)
            print('valid neg loss=', valid_avg_loss)
            if self.lr_decay:
                self.valid_neg_loss.append(valid_avg_loss)
                if valid_avg_loss < self.neg_min_loss:
                    self.neg_min_loss = valid_avg_loss
                if len(self.valid_loss) > 50:
                    if np.average(self.valid_loss[-25:]) - np.average(self.valid_loss[-50:-25]) < self.lr_decay_thres * self.lr_decay_coef and np.average(self.valid_neg_loss[-50:-25]) - np.average(self.valid_neg_loss[-25:]) < self.lr_decay_thres * self.lr_decay_coef:
                        self.lr_decay_coef /= 2
                print("lr=", self.lr * self.lr_decay_coef)
                print("pos_max_loss=", self.pos_max_loss,", neg_min_loss=", self.neg_min_loss)
            print('\n=========================\n')
        return this_loss

    def train1epoch_associative(self, sess, lr, epoch):
        num_AM_batch = int(self.multiG.num_align_desc() / self.batch_sizeA)
        if epoch <= 1:
            print('num_desc_batch =', num_AM_batch)
        #loss1, loss2 = self.train1epoch_AM(sess, num_AM_batch, lr, epoch)
        #return (loss1, loss2)
        loss1 = self.train1epoch_AM(sess, num_AM_batch, lr, epoch)
        return loss1

    def train_GRU(self, epochs=50, save_every_epoch=25, lr=0.001, m1=0.5):
        #sess = tf.Session()
        #sess.run(tf.initialize_all_variables())
        self.tf_parts._m1 = m1  
        t0 = time.time()
        self.lr = lr
        #try not initialize
        #self.initialize_encoders(self.sess, lr)
        for epoch in range(epochs):
            loss1 = self.train1epoch_associative(self.sess, lr, epoch)
            print("Time use: %d" % (time.time() - t0))
            if np.isnan(loss1):
                print("Training collapsed:",loss1)
                pdb.set_trace()
                return
            if self.valid and self.lr_decay_coef < self.lr_decay_stop:
                print("\n======XXXXXXXXX======\nTraining stopped at epoch",epoch,"by lr decay.\n======XXXXXXXXX======")
                break
            if (epoch + 1) % save_every_epoch == 0:
                this_save_path = self.tf_parts._saver.save(self.sess, self.save_path)
                #self.multiG.save(self.multiG_save_path)
                print("GRU saved in file: %s. Multi-graph saved in file: %s" % (this_save_path, self.multiG_save_path))
                print("\n=====================\nWe are not saving multiG as it is too large!!!\n======================")
        this_save_path = self.tf_parts._saver.save(self.sess, self.save_path)
        print("GRU saved in file: %s" % this_save_path)
        print("Done")

# A safer loading is available in Tester, with parameters like batch_size and dim recorded in the corresponding Data component
def load_tfparts(multiG, batch_sizeA=64,
                save_path = 'this-gru.ckpt'):
    tf_parts = gru_encoder.TFParts(
                          dim=multiG.desc_dim,
                          wv_dim1=self.multiG.KG1.wv_dim,
                          wv_dim2=self.multiG.KG2.wv_dim,
                          length1=self.multiG.KG1.desc_length,
                          length2=self.multiG.KG2.desc_length,
                          batch_sizeA=batch_sizeA,
                          learning_phase=True)
    #with tf.Session() as sess:
    sess = tf.Session()
    tf_parts._saver.restore(sess, save_path)