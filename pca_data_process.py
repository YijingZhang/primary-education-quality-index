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
    def pca(self,n_components = 0.999):
        data = np.loadtxt(self.filename).transpose()
        data = StandardScaler().fit_transform(data)
        pca_obj = sklearn.decomposition.PCA(n_components)
        pca_obj.fit(data)
        x = pca_obj.components_
        print('Original data shape is:{}'.format(data.shape))
        print('After PCA, the shape is:{}'.format(x.shape))
        print(x)
        return 0

if __name__ == '__main__':
    filename = 'data.txt'
    PCA(filename).pca()