#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'a demo python script'

import numpy as np
import tensorflow as tf
import random
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pca_data_process import PCA, Index
from Graph import Graph

lr = 0.01
epochs = 500
batch_size = 2
input_dim = 6
hidden_dim = [2, 1]
test_size = 5
sample_num = 31
data_dir = 'data.txt'
index_dir = 'index.txt'
corresponding_order_dir = 'corresponding_order.txt'
indices = list(range(sample_num))
random.shuffle(indices)
data = PCA(data_dir).pca()
label = MinMaxScaler().fit_transform(Index(index_dir, corresponding_order_dir).index_score()[...,np.newaxis])
test_data = data[indices[:test_size],:]
test_label = label[indices[:test_size],:]
train_data = data[indices[test_size:],:]
train_label = label[indices[test_size:],:]
Graph(input_dim,hidden_dim,lr,epochs,batch_size).train(train_data,train_label,test_data,test_label)
