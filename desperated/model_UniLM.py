# coding:utf-8
# Produced by Andysin Zhang
# 26_Sep_2019
# Inspired by 
# <<Unified Language Model Pre-training for Natural Language Understanding and Generation>>
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
import copy
import numpy as np
import tensorflow as tf
import model_helper as _mh

from model import BertModel
from hparams_config import config as config_
from log import log_info as _info
from log import log_error as _error

__all__ = ['UniLM']

class UniLM(BertModel):
    """"UniLM inherits from BERT model."""
    def __init__(self, config, is_training, scope=None):
        self.train_type = config.train_type
        self.output_length = tf.placeholder(tf.int32, name='output_length')
        # self.input_mask = tf.placeholder(tf.float32, [None, None, None], name='input_mask')
       
        super(UniLM, self).__init__(config, is_training, scope)
    
    def _create_attention_mask(self, batch_size):
        """Create mask for attention,
            train_type equals tp 1 is for left-to-right lm,
            otherwise, is for seq-seq lm.
        """
        mask = tf.cond(tf.equal(self.train_type, 1), 
                       lambda: self._create_attention_mask_LR_LM(batch_size),
                       lambda: self._create_attention_mask_Seq_Seq(batch_size))
        return mask
    
    def _create_attention_mask_LR_LM(self, batch_size):
        """create mask for left-to-right language model."""
        return tf.cast(self.input_mask, tf.float32)

    def _create_attention_mask_Seq_Seq(self, batch_size):
        # upper left
        length = self.input_length - self.output_length
        ul = tf.ones([length, length], dtype=tf.float32)
        ur = tf.zeros([length, self.output_length], dtype=tf.float32)
        bl = tf.ones([self.output_length, length], dtype=tf.float32)
        br = tf.cast(self.input_mask, tf.float32)

        ul_ur = tf.concat([ul, ur], axis=1)
        bl_br = tf.concat([bl, br], axis=1)
        mask = tf.concat([ul_ur, bl_br], axis=0)
        mask = tf.tile(tf.expand_dims(mask, 0), multiples=[batch_size, 1, 1])
        return mask
    
    def train(self, sess, data):
        assert self.is_training
        if self.pos_type is 'trigonometrical':
            feed = {self.input_ids: data.input_ids,
                    self.input_mask: data.input_mask,
                    self.input_length: data.input_length,
                    self.output_ids: data.output_ids,
                    self.positional_embeddings: data.positional_embeddings,
                    self.output_length: data.output_length}
        else:
            feed = {self.input_ids: data.input_ids,
                    self.input_mask: data.input_mask,
                    self.input_length: data.input_length,
                    self.output_ids: data.output_ids,
                    self.output_length: data.output_length}

        return sess.run([self.global_step, 
                         self.learning_rate, 
                         self.loss_bs,
                         self.upgrade,
                         self.logits,
                         self.train_summary], feed_dict=feed)