#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'a demo python script'

__author__ = 'Zhangyijing'
import numpy as np
import tensorflow as tf
import copy

class Graph(object):
    '''
    construct the graph. Train and test the graph
    '''
    def __init__(self, num_features, hidden_dim = [5,1],lr=0.01,epochs = 10,bacth_size = 4):
        self.hidden_dim = hidden_dim
        self.input_dim = num_features
        self.lr = lr
        self.epochs = epochs
        self.batch_size = bacth_size
    def _weight_variable(self,shape):
        return tf.get_variable('weight',shape,np.float32,tf.random_normal_initializer())
    def _bias_variable(self, shape):
        return tf.get_variable('bais',shape,np.float32,tf.random_normal_initializer())
    def build(self,input):
        input_dim = self.input_dim
        output = input
        self.hidden_dim = [input_dim]+self.hidden_dim
        for i in range(len(self.hidden_dim)-1):
            with tf.variable_scope('weight{}'.format(i)) as scope:
                weight = self._weight_variable([self.hidden_dim[i], self.hidden_dim[i+1]])
                bias = self._bias_variable([self.hidden_dim[i+1]])
                if i == len(self.hidden_dim) - 2:
                    output_logits = tf.matmul(output,weight) + bias
                    return output_logits
                output = tf.nn.sigmoid(tf.matmul(output,weight) + bias)
    def train(self,train_data,train_label,test_data,test_label):
        '''
        :param train_data: with shape (sample_nums, features_num)
        :param train_label: with shape (sample_num, 1)
        :return:
        '''
        x = tf.placeholder(dtype=tf.float32, shape=[None,self.input_dim],name='bath_input_tensor')
        y = tf.placeholder(dtype=tf.float32, shape=[None,self.hidden_dim[-1]], name='batch_output_tensor')
        output_logits = tf.sigmoid(self.build(x))
        loss = tf.nn.l2_loss(y - output_logits)
        train = tf.train.AdamOptimizer(self.lr).minimize(loss)
        batch_num = int(np.shape(train_data)[0] // self.batch_size)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(self.epochs):
                for batch in range(batch_num):
                    train_batch_data = train_data[self.batch_size*batch:self.batch_size*(batch+1),:]
                    train_batch_label = train_label[self.batch_size*batch:self.batch_size*(batch+1),:]
                    feed_dict = {x:train_batch_data, y:train_batch_label}
                    sess.run(train, feed_dict)
                print('epoch{},loss:{}'.format(epoch,sess.run(loss,feed_dict)))
            feed_dict = {x:test_data,y:test_label}
            evaluated_score = sess.run(output_logits,feed_dict)
            print(np.concatenate([evaluated_score,test_label],axis=1))





