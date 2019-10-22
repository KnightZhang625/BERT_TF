# coding:utf-8
# Produced by Andysin Zhang
# 22_Oct_2019
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
import tensorflow as tf
import model_helper as _mh

from pathlib import Path
PROJECT_PATH = Path(__file__).absolute().parent
sys.path.insert(0, str(PROJECT_PATH))

from utils.log import log_info as _info
from utils.log import log_error as _error

def tranformer_model(input_tensor,
                     attention_mask=None,
                     hidden_size=1024,
                     num_hidden_layers=12,
                     num_attention_heads=12,
                     intermediate_size=3072,
                     intermediate_act_fn=_mh.gelu,
                     hidden_dropout_prob=0.1,
                     attention_probs_dropout_prob=0.1,
                     initializer_range=0.02,
                     do_return_all_layers=False,
                     share_parameter_across_layers=True):
    """Multi-head, multi-layer Transformer.
    
    Args:
        input_tensor: float Tensor of shape [batch_size, seq_length, hidden_size].
        attention_mask: (optional) int32 Tensor of shape [batch_size, seq_length, seq_length],
            where 1 indicates the position can be attended and 0 indicates the position cannot be attended.
        hidden_size: int. Hidden size of the Transformer.
        num_hidden_layers: int. Number of layers in the Transformer.
        num_attention_heads: int. Number of attention heads in the Transformer.
        intermediate_size: int. The size of the feed forward layer.
        intermediate_act_fn: activation function after feed forward layer.
        hidden_dropout_prob: float.
        attention_probs_dropout_prob: float.
        initializer_range: float.
        do_return_all_layers: bool. Return the output from all the hidden layers or just the final layer.
        share_parameter_across_layers: bool. Whether share parameters across each attention layer.

    Returns:
        float Tensor of shape [batch_size, seq_length, hidden_size],
        or a list contains 'num_hidden_layers' float Tensor.
    """
    if hidden_size % num_attention_heads != 0:
        _error('The hidden size {} cannot be divided by the number of attention heads {}'.format(hidden_size, num_attention_heads))
        raise ValueError
    
    # the hidden size for each head
    attention_head_size = int(hidden_size / num_attention_heads)
    input_shape = _mh.get_shape_list(input_tensor, expected_rank=3)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    input_width = input_shape[2]
    