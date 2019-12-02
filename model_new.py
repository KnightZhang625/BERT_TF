# coding:utf-8
# Produced by Andysin Zhang
# 02_Dec_2019
# Rewrite the original classification code by Lee, 
# Appreciate for the wonderful work.
#
# Copyright 2019 TCL Inc. All Rights Reserverd.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import sys
import copy
import tensorflow as tf
import tensorflow.contrib.slim as slim

import model_helper as _mh
from utils.log import log_info as _info
from utils.log import log_error as _error

class TextCNN(object):
    def __init__(self, input_ids, is_training, config):
        _info('TextCNN Initialize...')
        
        # config
        config = copy.deepcopy(config)
        self.dropout_prob = config.dropout_prob
        if not is_training:
            self.dropout_prob = 0.0
        self.num_classes = config.num_classes

        # input        
        input_shape = _mh.get_shape_list(input_ids, expected_rank=2)
        batch_size = input_shape[0]

        # embedding
        with tf.variable_scope('embeddings'):
            self.embedding_table = tf.get_variable(
                name='embedding_table',
                shape=[config.vocab_size, config.embedding_size],
                initializer=_mh.create_initializer(config.initializer_range))

            # [b, s, e]
            self.embedding_output = tf.nn.embedding_lookup(
                self.embedding_table, input_ids)
            
            # I don't know why the original code expand dimension for embedding_output,
            # however, the expanded result is [b, s, e, 1]
            self.embedding_output_expanded = tf.expand_dims(self.embedding_output, -1)
            self.embedding_output_expanded = tf.reshape(self.embedding_output_expanded,
                                                        shape=[config.batch_size, config.max_length, config.embedding_size, 1])
  
        
        # the same as the original, except for variable names and some revisements.
        # Creat  4*( convolution + maxpool) layer
        with slim.arg_scope([slim.conv2d], padding='SAME',weights_initializer=tf.truncated_normal_initializer(stddev=0.05), data_format='NHWC'):
            net_1 = slim.repeat(self.embedding_output_expanded, 1,slim.conv2d, 128, [11, 11], scope='conv1')
            net_1 = slim.max_pool2d(net_1, [2,2], scope='pool1')
            net_1 = tf.nn.dropout(net_1, keep_prob=self.dropout_prob)

            net_2 = slim.repeat(net_1, 1, slim.conv2d, 128, [7,7], scope='conv2')
            net_2 = slim.max_pool2d(net_2, [2, 2], scope='pool2')
            net_2 = tf.nn.dropout(net_2, keep_prob=self.dropout_prob)

            net_3 = slim.repeat(net_2, 1, slim.conv2d, 64, [5,5], scope='conv3')
            net_3 = slim.max_pool2d(net_3, [2, 2], scope='pool3')
            net_3 = tf.nn.dropout(net_3, keep_prob=self.dropout_prob)

            net_4 = slim.repeat(net_3, 1, slim.conv2d, 64, [3,3], scope='conv4')
            net_4 = slim.max_pool2d(net_4,[2,2],scope='pool4')
            net_4 = tf.nn.dropout(net_4, keep_prob=self.dropout_prob)
            net_4_shape = _mh.get_shape_list(net_4, expected_rank=4)
            
        # the same as the original, except for adding some spaces between the code.
        # Creat 3*FC layer
        with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu,
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.05)):
            self.fc_input_x = tf.reshape(net_4, shape=[-1, net_4_shape[1] * net_4_shape[2] * net_4_shape[3]])
            self.fc_input_x = tf.reshape(net_4, shape=[config.batch_size, -1])
            fc_net = tf.layers.dense(self.fc_input_x, 1024,
                                     name='fc3',
                                     kernel_initializer=_mh.create_initializer(0.2))
            fc_net = slim.dropout(fc_net, keep_prob=self.dropout_prob, scope='fc_drop1')
            fc_net = tf.layers.dense(fc_net, 1024, 
                                      name='fc32', 
                                      kernel_initializer=_mh.create_initializer(0.2))
        
        # the same as the original, however, I would like to put this into train.
        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):

            W = tf.get_variable(
                "W",
                shape=[1024, self.num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name="b")

            self.scores = tf.nn.xw_plus_b(fc_net, W, b, name="scores")
            self.softmax_data = tf.nn.softmax(self.scores,name="cf_softmax")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
    
    # the new code
    def get_output(self):
        return self.scores