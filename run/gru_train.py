
# coding: utf-8

# In[1]:



# In[2]:

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# In[3]:
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

import numpy as np
import os
import tensorflow as tf

from KG import KG
from multiG import multiG   # we don't import individual things in a model. This is to make auto reloading in Notebook happy
import gru_encoder
import learner_gru
from learner_gru import Trainer


# In[4]:
model_path = './test-GRU.ckpt'
data_path = 'test-multiG2.bin'
kgf1 = '../preprocess/60k/en_60k.csv'
kgf2 = '../preprocess/60k/fr_60k.csv'
alignf = '../preprocess/60k/prop/en_fr_60k_round1_KGbT_train35.csv'#'../preprocess/60k/prop/en_fr_60k_round1_KGbT_test68.csv'#'../preprocess/60k/prop/en_fr_60k_sanity_check_256.csv'
ebf1 = '../embed/sub5000000.200.envec.txt'
ebf2 = '../embed/sub5000000.200.frvec.txt'
validf = '../preprocess/60k/prop/en_fr_60k_round1_KGbT_test68.csv'
descf1 = '../embed/en_60k_short.tknzd.txt'#'../embed/en_60k_short_desc.tknzd.txt'
descf2 = '../embed/fr_60k_short.tknzd.txt'#'../embed/fr_60k_short_desc.tknzd.txt'
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
this_data.load_valid(validf, size=2048)

# In[ ]:

m_train = Trainer()
#dim=64, batch_sizeA=128, m1=1., save_path = 'this-gru.ckpt', multiG_save_path = 'this-multiG2.bin'
m_train.build(this_data, dim=100, batch_sizeA=256, save_path = model_path, multiG_save_path = data_path, L1=False)


# In[ ]:

m_train.train_GRU( epochs=2000, save_every_epoch=20, lr=0.001)

# In[ ]:



