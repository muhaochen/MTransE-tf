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
if not os.path.exists('../results/try_gru'):
    os.makedirs('../results/try_gru')

import numpy as np
from numpy import linalg as LA
import tensorflow as tf
import time

import multiG
import tester_gru
from tester_gru import Tester
from KG import KG
from multiG import multiG

model_file = os.path.join(os.path.dirname(__file__), 'test-gru.ckpt')
data_file = 'test-multiG.bin'
test_data = '../preprocess/60k/prop/en_fr_60k_round1_KGbT_test68.csv'#'../preprocess/60k/prop/en_fr_60k_round1_KGbT_test68.csv'#'../preprocess/60k/prop/en_fr_60k_round1_KGbT_train32.csv'
result_file = '../results/try_gru/test_result.txt'
result_file2 = '../results/try_gru/test_recall.txt'
prop_file = '../preprocess/60k/prop/txt_pback_KG_r1.csv'
pct = 0.11

fp = open(result_file, 'w')
fp2 = open(result_file2, 'w')
fp3 = open(prop_file, 'w')

kgf1 = '../preprocess/60k/en_60k.csv'
kgf2 = '../preprocess/60k/fr_60k.csv'
alignf = '../preprocess/60k/prop/en_fr_60k_round1_KGbT_train32.csv'#'../preprocess/60k/prop/en_fr_60k_round1_KGbT_test68.csv'#'../preprocess/60k/prop/en_fr_60k_sanity_check_256.csv'#'../preprocess/60k/prop/en_fr_60k_round1_KGbT_train32.csv'
ebf1 = '../embed/sub5000000.200.envec.txt'
ebf2 = '../embed/sub5000000.200.frvec.txt'
descf1 = '../embed/en_60k_two_sen_desc.tknzd.txt'#'../embed/en_60k_short_desc.tknzd.txt'
descf2 = '../embed/fr_60k_two_sen_trans.tknzd.txt'#'../embed/fr_60k_short_desc.tknzd.txt'
entf1 = '../embed/en_60k_short_entity.txt'
entf2 = '../embed/fr_60k_short_entity.txt'

KG1 = KG()
KG2 = KG()
en_stopwords = KG1.load_stop_words('../embed/stop_words_en.txt')
fr_stopwords = KG2.load_stop_words('../embed/stop_words_fr.txt')
KG1.load_triples(filename = kgf1, splitter = '@@@', line_end = '\n')
KG2.load_triples(filename = kgf2, splitter = '@@@', line_end = '\n')
KG1.load_word2vec(ebf1)
KG1.load_descriptions(entf1, descf1, desc_length = 36, lower=False, stop_words=en_stopwords, padding_front=True)
KG2.load_word2vec(ebf2)
KG2.load_descriptions(entf2, descf2, desc_length = 36, lower=False, stop_words=fr_stopwords, padding_front=True)
this_data = multiG(KG1, KG2)
this_data.load_align(filename = alignf, desc=True, lan1 = 'en', lan2 = 'fr', splitter = '@@@', line_end = '\n')

tester = Tester()
tester.build(this_data, save_path = 'test-GRU.ckpt', dim=100, batch_sizeA=256, L1=False, bybatch=True)
tester.load_test_data(test_data, splitter = '@@@', line_end = '\n')

desc_record1 = set(KG1.desc_index)
desc_record2 = set(KG2.desc_index)

import multiprocessing
from multiprocessing import Process, Value, Lock, Manager

cpu_count = multiprocessing.cpu_count()

manager = Manager()

index = Value('i', 0, lock=True) #index
rst_predict = manager.list() #scores for each case

t0 = time.time()

#scan_index = tester.multiG.aligned_KG2_index
scan_index = tester.search_space_r
#What if desc_record2 is not used?
def test(tester, index, desc_record1, desc_record2, rst_predict):
    while index.value < len(tester.test_align):
        id = index.value
        index.value += 1
        if id % 100 == 0:
            print("Tested %d in %d seconds." % (id+1, time.time()-t0))
        e1, e2 = tester.test_align[id]
        if e1 not in desc_record1 or e2 not in desc_record2:
            continue
        vec_e1 = tester.ent_index2vec(e1, source = 1)
        vec_e2 = tester.ent_index2vec(e2, source = 2)
        #vec_proj_e1 = tester.projection(e1, source = 1)
        vec_pool_e2 = tester.vec_e[2]
        rst = tester.NN(vec_e1, vec_pool_e2, scan_index)
        hit = 0.
        if rst[0] == e2 or rst[0] in tester.lr_map[e1]:
            hit = 1.
        rank = tester.rank_index_from(vec_e1, vec_pool_e2, e2, scan_index, self_id = None)
        dist = tester.dist_from(vec_e1, vec_e2)
        rst_predict.append((rst[0], hit, rst[1], dist, rank, e1, e2, tester.ent_index2str(e1, source=1), tester.ent_index2str(rst[0], source=2), tester.ent_index2str(e2, source=2)))

print("===\n===\nRemember to change above scan_index back!\n===\n===")
processes = [Process(target=test, args=(tester, index, desc_record1, desc_record2, rst_predict)) for x in range(cpu_count - 1)]
for p in processes:
    p.start()
for p in processes:
    p.join()

list1 = [x for x in rst_predict]
list1.sort(key=lambda x: x[2], reverse=True)
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
    writeln = str(line[7]) + '@@@' + str(line[8]) + '\n'
    if writeln not in check_dup:
        fp3.write(writeln)
        check_dup.add(writeln)
    this_count += 1
    if this_count > count*pct:
        break

fp.close()
fp2.close()
fp3.close()
