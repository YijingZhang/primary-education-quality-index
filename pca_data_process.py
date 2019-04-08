#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'a demo python script'

__author__ = 'Zhangyijing'

import numpy as np
import sklearn.decomposition
from sklearn.preprocessing import StandardScaler

class PCA(object):
    def __init__(self,filename):
        self.filename = filename
    def pca(self,n_components = 0.95):
        data = np.loadtxt(self.filename)
        data = StandardScaler().fit_transform(data)
        pca_obj = sklearn.decomposition.PCA(n_components)
        pca_obj.fit(data)
        x = pca_obj.transform(data)
        print('Original data shape is:{}'.format(data.shape))
        print('After PCA, the shape is:{}'.format(x.shape))
        # x = sklearn.preprocessing.scale(x)
        # print(x)
        return x
class Index(object):
    def __init__(self,index_name,order_name):
        self.index_name = index_name
        self.order_name = order_name
    def index_score(self):
        order = list(map(int,np.loadtxt(self.order_name)))
        index = np.loadtxt(self.index_name)
        index = index[order]
        # print(index)
        return index


if __name__ == '__main__':
    filename = 'data.txt'
    PCA(filename).pca()
    order_name = 'corresponding_order.txt'
    index_name = 'index.txt'
    index = Index(index_name, order_name).index_score()