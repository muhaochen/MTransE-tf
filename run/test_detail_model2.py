from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))


import os
if not os.path.exists('../results'):
    os.makedirs('../results')

import os
if not os.path.exists('../results/detail_m2'):
    os.makedirs('../results/detail_m2')

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
from numpy import linalg as LA
import tensorflow as tf
import time
import multiG  
import model2 as model
from tester_MTransE2 import Tester

model_file = 'test-model-m2.ckpt'
data_file = 'test-multiG-m2.bin'
test_data = '../preprocess/60k/en_fr_60k_test75.csv'
except_data = '../preprocess/60k/en_fr_60k_train25.csv'
old_data = '../preprocess/60k/en_fr_60k_train25.csv'
result_file = '../results/detail/detail_result_m2.txt'
result_file2 = '../results/detail/detail_recall_m2.txt'

topK = 20
fp = open(result_file, 'w')
fp2 = open(result_file2, 'w')

max_check = 4000

dup_set = set([])
for line in open(old_data):
    dup_set.add(line.rstrip().split('@@@')[0])

tester = Tester()
tester.build(save_path = model_file, data_save_path = data_file)
tester.load_test_data(test_data, splitter = '@@@', line_end = '\n')
tester.load_except_data(except_data, splitter = '@@@', line_end = '\n')

import multiprocessing
from multiprocessing import Process, Value, Lock, Manager

cpu_count = multiprocessing.cpu_count()

manager = Manager()

index = Value('i', 0, lock=True) #index
rst_predict = manager.list() #scores for each case
rst_record = manager.list()
rank_record = manager.list()
prop_record = manager.list()
t0 = time.time()

def test(tester, index, rst_predict, rst_record, rank_record, prop_record):
    while index.value < len(tester.test_align):
        id = index.value
        index.value += 1
        if id % 100 == 0:
            print("Tested %d in %d seconds." % (id+1, time.time()-t0))
            try:
                print(np.mean(rst_predict, axis=0))
            except:
                pass
        e1, e2 = tester.test_align[id]
        #vec_e1 = tester.ent_index2vec(e1, source = 1)
        vec_proj_e1 = tester.projection(e1, source = 1)
        vec_pool_e2 = tester.vec_e[2]
        rst = tester.kNN(vec_proj_e1, vec_pool_e2, topK)#, except_ids=tester.aligned[2])
        this_hit = []
        hit = 0.
        strl = tester.ent_index2str(rst[0][0], 2)
        strr = tester.ent_index2str(e2, 2)
        this_index = 0
        this_rank = None
        for pr in rst:
            this_index += 1
            if pr[0] == e2 or pr[0] in tester.lr_map[e1] or (hit < 1. and tester.ent_index2str(pr[0], 2) == strr):
                hit = 1.
                rst_record.append((1., pr[1]))
                this_rank = this_index
            else:
                rst_record.append((0., pr[1]))
            this_hit.append(hit)
        hit_first = 0
        if rst[0][0] == e2 or rst[0][0] in tester.lr_map[e1] or strl == strr:
            hit_first = 1
        if this_rank is None:
            this_rank = tester.rank_index_from(vec_proj_e1, vec_pool_e2, e2, except_ids=tester.aligned[2])
        if this_rank > max_check:
            continue
        rst_predict.append(np.array(this_hit))
        rank_record.append(1.0 / (1.0 * this_rank))
        prop_record.append((hit_first, rst[0][1], strl, strr))

# tester.rel_num_cases
processes = [Process(target=test, args=(tester, index, rst_predict, rst_record, rank_record, prop_record)) for x in range(cpu_count - 1)]
for p in processes:
    p.start()
for p in processes:
    p.join()

mean_rank = np.mean(rank_record)
hits = np.mean(rst_predict, axis=0)

fp.write(str(mean_rank)+'\n')
fp.write(' '.join([str(x) for x in hits]) + '\n')

list1 = [x for x in rst_record]
list1.sort(key=lambda x: x[1])
total = 0
for line in list1:
    fp2.write('\t'.join([str(x) for x in line]) + '\n')
    if line[0] == 1.:
        total += 1
    if total >= index.value:
        break

fp.close()
fp2.close()