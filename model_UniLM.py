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

class UniLM(BertModel):
    """"UniLM inherits from BERT model."""
    def __init__(self, config, is_training, scope=None):
        super(UniLM, self).__init__(config, is_training, scope)

        self.train_type = tf.placeholder(tf.int32, name='train_type')
        self.output_length = tf.placeholder(tf.int32, name='output_length')
        self.lr_mask = tf.placeholder(tf.float32, [None, None, None], name='lr_mask')
    
    def _create_attention_mask(self, batch_size):
        """Create mask for attention,
            train_type equals tp 1 is for left-to-right lm,
            otherwise, is for seq-seq lm.
        """
        mask = tf.cond(tf.equal(self.train_type, 1), 
                       lambda: self._create_attention_mask_LR_LM(batch_size),
                       lambda: self._create_attention_mask_Seq_Seq(batch_size))
    
    def _create_attention_mask_LR_LM(self, batch_size):
        """create mask for left-to-right language model."""
        return self.lr_mask

    def _create_attention_mask_Seq_Seq(self, batch_size):
        # upper left
        ul = np.zeros([self.a, self.input_length], )