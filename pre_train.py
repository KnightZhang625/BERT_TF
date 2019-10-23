# coding:utf-8
# Produced by Andysin Zhang
# 23_Oct_2019
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

""""Run masked LM/next sentence masked_lm pre-training for ALBERT."""

import sys
import tensorflow as tf
# tf.enable_eager_execution()

from pathlib import Path
PROJECT_PATH = Path(__file__).absolute().parent
sys.path.insert(0, str(PROJECT_PATH))

from utils.setup import Setup
setup = Setup()

from model import BertModel
from utils.log import log_info as _info
from utils.log import log_error as _error

flags = tf.flags
FLAGS = flags.FLAGS

# Prototype for tf.estimator
def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, 
                     use_one_hot_embeddings):
    """Returns 'model_fn' closure for Estomator,
       use closure is because of building the model requires
       some paramters, sending them into the 'params' is not a good deal."""

    def model_fn(features, labels, mode, params):
        """this is prototype syntax, all parameters are necessary."""
        # obtain the data
        _info('*** Features ***')
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features['input_ids']       # [batch_size, seq_length]
        input_mask = features['input_mask']     # [batch_size, seq_length]
        # segment_idx = features['segment_dis']
        masked_lm_positions = features['masked_lm_positions']   # [batch_size, seq_length], specify the answer
        masked_lm_ids = features['masked_lm_ids']               # [batch_size, answer_seq_length], specify the answer labels
        masked_lm_weights = features['maked_lm_weights']        # [batch_size, seq_length], [1, 1, 0], 0 refers to the mask
        # next_sentence_labels = features['next_sentence_labels']