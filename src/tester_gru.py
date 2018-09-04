''' Module for held-out test.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from numpy import linalg as LA
import heapq as HP
import sys

import multiG
from multiG import multiG
import KG
from KG import KG
import gru_encoder
import trainer

# This class is used to load and combine a TF_Parts and a Data object, and provides some useful methods for training
class Tester(object):
    def __init__(self):
        self.tf_parts = None
        self.multiG = None
        self.vec_e = {}
        # below for test data
        self.test_align = np.array([0])
        # L1 to L2 map
        self.lr_map = {}
        # L2 to L1 map
        self.rl_map = {}
        self.search_space_l = np.array([0])
        self.search_space_r = np.array([0])
        self.sess = None
    
    def build(self, multiG, 
                 save_path = 'test-GRU.ckpt',
                 dim=64, 
                 batch_sizeA=128, 
                 L1=False, bybatch=False):
        self.multiG = multiG
        tf.reset_default_graph()
        self.tf_parts = gru_encoder.TFParts(
                          dim=dim,
                          wv_dim1=self.multiG.KG1.wv_dim,
                          wv_dim2=self.multiG.KG2.wv_dim,
                          length1=self.multiG.KG1.desc_length,
                          length2=self.multiG.KG2.desc_length,
                          batch_sizeA=batch_sizeA,
                          learning_phase=False)
        self.sess = sess = tf.Session()
        self.tf_parts._saver.restore(sess, save_path)  # load it
        if not bybatch:
            value_ht1, value_ht2 = sess.run([self.tf_parts._dump_embed1, self.tf_parts._dump_embed2],
                        feed_dict={self.tf_parts._desc1: self.multiG.KG1.desc_embed_padded[self.multiG.KG1.desc_index],
                                   self.tf_parts._desc2: self.multiG.KG2.desc_embed_padded[self.multiG.KG2.desc_index]})  # extract values.
        else:
            value_ht1 = sess.run([self.tf_parts._dump_embed1],
                        feed_dict={self.tf_parts._desc1: [self.multiG.KG1.desc_embed_padded[self.multiG.KG1.desc_index[0]]]})
            value_ht2 = sess.run([self.tf_parts._dump_embed2],
                        feed_dict={self.tf_parts._desc2: [self.multiG.KG2.desc_embed_padded[self.multiG.KG2.desc_index[0]]]})
            for i in range(1, len(self.multiG.KG1.desc_index)):
                value_ht1 = np.concatenate((value_ht1, sess.run([self.tf_parts._dump_embed1],
                        feed_dict={self.tf_parts._desc1: [self.multiG.KG1.desc_embed_padded[self.multiG.KG1.desc_index[i]]]})))
            for i in range(1, len(self.multiG.KG2.desc_index)):
                value_ht2 = np.concatenate((value_ht2, sess.run([self.tf_parts._dump_embed2],
                        feed_dict={self.tf_parts._desc2: [self.multiG.KG2.desc_embed_padded[self.multiG.KG2.desc_index[i]]]})))
        #sess.close()
        temp_vec_e1 = np.array(value_ht1)
        temp_vec_e2 = np.array(value_ht2)
        print('temp_vec_e1',temp_vec_e1.shape)
        #ordered embedding array that is zero-padded
        self.vec_e[1] = np.zeros((self.multiG.KG1.num_ents(), dim))
        self.vec_e[2] = np.zeros((self.multiG.KG2.num_ents(), dim))
        for i in range(len(self.multiG.KG1.desc_index)):
            self.vec_e[1][self.multiG.KG1.desc_index[i]] = temp_vec_e1[i]
        for i in range(len(self.multiG.KG2.desc_index)):
            self.vec_e[2][self.multiG.KG2.desc_index[i]] = temp_vec_e2[i]
        print("\n==================\nEmbedded all descriptions:", len(self.vec_e[1]), len(self.vec_e[2]), ", remember to use desc_index of each KG.\n===================\n")
    
    def desc_embed_to_vec(self, desc_embed_list, source=1, bybatch=False):
        assert(source in [1, 2])
        sess = self.sess
        value_ht = None
        target = self.tf_parts._dump_embed1
        place_holder = self.tf_parts._desc1
        if source == 2:
            target = self.tf_parts._dump_embed2
            place_holder = self.tf_parts._desc2
        if not bybatch:
            value_ht = sess.run([target],
                        feed_dict={place_holder: desc_embed_list})  # extract values.
        else:
            value_ht = sess.run([target],
                        feed_dict={place_holder: [desc_embed_list[0]]})
            for i in range(1, len(desc_embed_list)):
                value_ht = np.concatenate((value_ht, sess.run([target],
                        feed_dict={place_holder: [desc_embed_list[i]]})))
        return np.array(value_ht[0])

    def load_test_data(self, filename, splitter = '@@@', line_end = '\n'):
        num_lines = 0
        align = []
        self.search_space_l = set([])
        self.search_space_r = set([])
        for line in open(filename):
            line = line.rstrip(line_end).split(splitter)
            if len(line) != 2:
                continue
            num_lines += 1
            e1 = self.multiG.KG1.ent_str2index(line[0])
            e2 = self.multiG.KG2.ent_str2index(line[1])
            if e1 == None or e2 == None:
                continue
            align.append([e1, e2])
            self.search_space_l.add(e1)
            self.search_space_r.add(e2)
            if self.lr_map.get(e1) == None:
                self.lr_map[e1] = set([e2])
            else:
                self.lr_map[e1].add(e2)
            if self.rl_map.get(e2) == None:
                self.rl_map[e2] = set([e1])
            else:
                self.rl_map[e2].add(e1)
        self.test_align = np.array(align, dtype=np.int32)
        self.search_space_l = np.array([e for e in self.search_space_l])
        self.search_space_r = np.array([e for e in self.search_space_r])
        print("Loaded test data from %s, %d out of %d." % (filename, len(align), num_lines))
    
    def ent_index2vec(self, e, source):
        assert (source in set([1, 2]))
        return self.vec_e[source][int(e)]

    def ent_str2vec(self, str, source):
        assert (source in set([1, 2]))
        KG = None
        if source == 1:
            KG = self.multiG.KG1
        else:
            KG = self.multiG.KG2
        this_index = KG.ent_str2index(str)
        if this_index == None:
            return None
        return self.vec_e[source][this_index]
    
    class index_dist:
        def __init__(self, index, dist):
            self.dist = dist
            self.index = index
            return
        def __lt__(self, other):
            return self.dist < other.dist
                
    def ent_index2str(self, id, source):
        KG = None
        if source == 1:
            KG = self.multiG.KG1
        else:
            KG = self.multiG.KG2
        return KG.ent_index2str(id)
    
    # input must contain a pool of vecs. return a list of indices and dist
    def kNN(self, vec, vec_pool, desc_index, topk=10, self_id=None):
        q = []
        for i in desc_index:
            #skip self
            if i == self_id:
                continue
            dist = np.dot(vec, vec_pool[i])
            if len(q) < topk:
                HP.heappush(q, self.index_dist(i, dist))
            else:
                tmp = HP.nsmallest(1, q)[0]
                if tmp.dist < dist:
                    HP.heapreplace(q, self.index_dist(i, dist) )
        rst = []
        while len(q) > 0:
            item = HP.heappop(q)
            rst.insert(0, (item.index, item.dist))
        return rst

    # input must contain a pool of vecs. return a list of indices and dist
    def NN(self, vec, vec_pool, desc_index, self_id=None):
        max_dist = float('-inf')
        rst = None
        for i in desc_index:
            #skip self
            if i == self_id:
                continue
            dist = np.dot(vec, vec_pool[i])
            if dist > max_dist:
                max_dist = dist
                rst = i
        return (rst, max_dist)
        
    # input must contain a pool of vecs. return a list of indices and dist. rank an index in a vec_pool from 
    def rank_index_from(self, vec, vec_pool, index, desc_index, self_id = None):
        dist = np.dot(vec, vec_pool[index])
        rank = 1
        for i in desc_index:
            if i == index or i == self_id:
                continue
            if dist < np.dot(vec, vec_pool[i]):
                rank += 1
        return rank
    
    def dist_from(self, vec, vec2):
        return np.dot(vec, vec2)
