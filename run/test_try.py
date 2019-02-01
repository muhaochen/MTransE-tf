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
if not os.path.exists('../results/try'):
    os.makedirs('../results/try')

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
from numpy import linalg as LA
import tensorflow as tf
import time

import multiG  
import model2 as model
from tester_MTransE2 import Tester

model_file = 'test-model.ckpt'
data_file = 'test-multiG.bin'
test_data = '../preprocess/60k/en_fr_60k_test75.csv'
except_data = '../preprocess/60k/en_fr_60k_train25.csv'
result_file = '../results/try/test_result.txt'
result_file2 = '../results/try/test_recall.txt'
prop_file = '../preprocess/60k/prop/KG_p_txt_r1_change.csv'
pct = 0.11

fp = open(result_file, 'w')
fp2 = open(result_file2, 'w')
fp3 = open(prop_file, 'w')

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

t0 = time.time()

def test(tester, index, rst_predict):
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
        hit = 0.
        if tester.ent_index2str(rst[0], 1) is None:
            continue
        if rst[0] == e2 or rst[0] in tester.lr_map[e1] or tester.ent_index2str(rst[0], 2) == tester.ent_index2str(e2, 2):
            hit = 1.
        rst_predict.append((rst[0], hit, rst[1], e1, e2, tester.ent_index2str(e1, source=1), tester.ent_index2str(rst[0], source=2), tester.ent_index2str(e2, source=2)))

# tester.rel_num_cases
processes = [Process(target=test, args=(tester, index, rst_predict)) for x in range(cpu_count - 1)]
for p in processes:
    p.start()
for p in processes:
    p.join()

list1 = [x for x in rst_predict]
list1.sort(key=lambda x: x[2])
total = 0.
for line in list1:
    total += 1
    fp.write('\t'.join([str(x) for x in line]) + '\n')

prec = 0.
rec = 0.
count = 0.
for line in list1:
    hit = int(line[1])
    prec = ( prec * count + hit ) / (count + 1)
    count += 1
    rec = count / total
    fp2.write(str(prec) + '\t' + str(rec) + '\t' + str(hit) + '\n')

this_count = 0
check_dup = set([])
for line in list1:
    writeln = str(line[5]) + '@@@' + str(line[6]) + '\n'
    if writeln not in check_dup:
        fp3.write(writeln)
        check_dup.add(writeln)
    this_count += 1
    if this_count > count*pct:
        break

fp.close()
fp2.close()
fp3.close()