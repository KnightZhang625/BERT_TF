# coding:utf-8
# Produced by Andysin Zhang
# 19_Oct_2019
# Inspired By the original Bert, Appreciate for the wonderful work
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
import model_helper as _mh

from pathlib import Path
PROJECT_PATH = Path(__file__).absolute().parent
sys.path.insert(0, str(PROJECT_PATH))

from utils.setup import Setup
setup = Setup()

from utils.log import log_info as _info
from utils.log import log_error as _error

import tensorflow as tf
# tf.enable_eager_execution()

class BertModel(object):
    """A Lite Bert Model"""
    def __init__(self, 
                 config, 
                 is_training, 
                 input_ids, 
                 input_mask=None, 
                 token_type_ids=None, 
                 use_one_hot_embeddings=False, 
                 scope=None):
        """"Constructor for ALBert.
        
        Args:
            config: # TODO
            is_training: bool. If True, enable dropout, else disable dropout.
            input_ids: int32 Tensor of shape [batch_size, seq_length].
            input_mask: (optional) int32 Tensor, 
                this is the mask for point the padding indices, [batch_size, seq_length].
            token_type_ids: (optional) int32 Tensor, point the words belonging to different segments, 
                [batch_size,seq_length].
            use_one_hot_embeddings: (optional) bool. Whether to use one-hot word embeddings 
                or tf.embedding_lookup() for the word embeddings.
        """
        config = copy.deepcopy(config)
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0

        input_shape = _mh.get_shape_list(input_ids, expected_rank=2)
        batch_size = input_shape[0]
        seq_length = input_shape[1]

        if input_mask is None:
            # each word is the real word, no padding.
            input_shape = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)

        if token_type_ids is None:
            token_type_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)
        
        with tf.variable_scope(scope, default_name='albert'):
            with tf.variable_scope('embeddings'):
                # 1. obtain embeddings
                self.embedding_output, self.embedding_table, self.projection_table = _mh.embedding_lookup_factorized(
                    input_ids=input_ids,
                    vocab_size=config.vocab_size,
                    hidden_size=config.hidden_size,
                    embedding_size=config.embedding_size,
                    use_one_hot_embedding=use_one_hot_embedding,
                    initializer_range=config.initializer_range,
                    word_embedding_name='word_embeddings')