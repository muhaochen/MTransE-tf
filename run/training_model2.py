
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

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
import tensorflow as tf

from KG import KG
from multiG import multiG   # we don't import individual things in a model. This is to make auto reloading in Notebook happy
import model2 as model
from trainer2 import Trainer


# In[4]:
model_path = './test-model-m2.ckpt'
data_path = 'test-multiG-m2.bin'
kgf1 = '../preprocess/60k/en_60k.csv'
kgf2 = '../preprocess/60k/fr_60k.csv'
alignf = '../preprocess/60k/en_fr_60k_train25.csv'

this_dim = 50

if len(sys.argv) > 1:
    this_dim = int(sys.argv[1])
    model_path = sys.argv[2]
    data_path = sys.argv[3]
    kgf1 = sys.argv[4]
    kgf2 = sys.argv[5]
    alignf = sys.argv[6]

KG1 = KG()
KG2 = KG()
KG1.load_triples(filename = kgf1, splitter = '@@@', line_end = '\n')
KG2.load_triples(filename = kgf2, splitter = '@@@', line_end = '\n')
this_data = multiG(KG1, KG2)
this_data.load_align(filename = alignf, lan1 = 'en', lan2 = 'fr', splitter = '@@@', line_end = '\n')



# In[ ]:

m_train = Trainer()
m_train.build(this_data, dim=this_dim, batch_sizeK=128, batch_sizeA=64, a1=5., a2=0.5, m1=0.5, save_path = model_path, multiG_save_path = data_path, L1=False)


# In[ ]:

m_train.train_MTransE( epochs=100, save_every_epoch=100, lr=0.001, a1=2.5, a2=0.5, m1=0.5, AM_fold=5, half_loss_per_epoch=150)


# In[ ]:



