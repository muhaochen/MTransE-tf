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

model_file = 'test-model-r2-m2.ckpt'
data_file = 'test-multiG-r2-m2.bin'
test_data = '../preprocess/60k/prop/en_fr_60k_round2_KGbT_test60.csv'
except_data = '../preprocess/60k/prop/en_fr_60k_round2_KGbT_train40.csv'
old_data = '../preprocess/60k/prop/en_fr_60k_round2_KGbT_train40.csv'
result_file3 = '../results/detail_m2/detail_prop_r2_m2.txt'
prop_file = '../preprocess/60k/prop/KG_p_txt_r2_m2_raw.csv'
topK = 1
pct = 0.11

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
        e1, e2 = tester.test_align[id]
        #vec_e1 = tester.ent_index2vec(e1, source = 1)
        vec_proj_e1 = tester.projection(e1, source = 1)
        vec_pool_e2 = tester.vec_e[2]
        rst = tester.NN(vec_proj_e1, vec_pool_e2, except_ids=tester.aligned[2])
        this_hit = []
        hit = 0.
        strl = tester.ent_index2str(rst[0], 2)
        if strl is None:
            continue
        strr = tester.ent_index2str(e2, 2)
        hit_first = 0
        if rst[0] == e2 or rst[0] in tester.lr_map[e1] or strl == strr:
            hit_first = 1
        prop_record.append((hit_first, rst[1], strl, strr))

# tester.rel_num_cases
processes = [Process(target=test, args=(tester, index, rst_predict, rst_record, rank_record, prop_record)) for x in range(cpu_count - 1)]
for p in processes:
    p.start()
for p in processes:
    p.join()

#add
fp3 = open(result_file3, 'w')
fp4 = open(prop_file, 'w')

list2 = [x for x in prop_record]
list2.sort(key=lambda x: x[1])
list3 = []
count = 0.
accuracy = 0.
recall = 0.
for line in list2:
    accuracy = (accuracy * count + line[0]) / (count + 1)
    count += 1
    recall = count / len(list2)
    list3.append((accuracy, recall, line[2], line[3]))

total = 0
for line in list3:
    fp3.write('\t'.join([str(x) for x in line]) + '\n')
    total += 1
    if total <= pct * len(list3):
        write_back = line[2]+'@@@'+line[3]
        if not (line[2] in dup_set):
            fp4.write(write_back + '\n')
fp3.close()
fp4.close()