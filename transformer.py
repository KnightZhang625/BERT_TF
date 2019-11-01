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
import math
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

    # residual layer need to perform on the outputs from all layers,
    # so the hidden size, i.e. the outputs from the transformer blocks
    # should be the same as the input_width, at the beginning, it is input tensor,
    # diffetentiate hidden_size from the intermediate_size,
    # intermediate layer is before the hidden layer.
    if input_width != hidden_size:
        _error('The width of the input tensor {} not not equal to the hidden size {}'.format(input_width, hidden_size))
        raise ValueError

    # create a list to save the output from each transformer layer]
    prev_output = input_tensor      # [batch_size, seq_length, width]
    all_layer_outputs = []
    for layer_idx in range(num_hidden_layers):
        if share_parameter_across_layers:
            name_variable_scope = 'layer_shared'
        else:
            name_variable_scope = 'layer_{}'.format*layer_idx
        
        # share the parameter across layers when share_parameter_across_layers us True and not the first layer
        with tf.variable_scope(name_variable_scope, reuse=True if (share_parameter_across_layers and layer_idx > 0) else False):
            layer_input = prev_output
            with tf.variable_scope('attention'):
                attention_heads = []
                with tf.variable_scope('self'):
                    attention_head = self_attention_layer(from_tensor=layer_input,
                                                          to_tensor=layer_input,
                                                          attention_mask=attention_mask,
                                                          num_attention_heads=num_attention_heads,
                                                          size_per_head=attention_head_size,
                                                          attention_probs_dropout_prob=attention_probs_dropout_prob,
                                                          initializer_range=initializer_range,
                                                          batch_size=batch_size,
                                                          from_seq_length=seq_length,
                                                          to_seq_length=seq_length)
                attention_output = attention_head
                # perform residual layer to finish the self-attention block
                with tf.variable_scope('output'):
                    attention_output = tf.layers.dense(
                        attention_output,
                        hidden_size,
                        kernel_initializer=_mh.create_initializer(initializer_range))
                    attention_output = _mh.dropout(attention_output, hidden_dropout_prob)
                    attention_output = _mh.layer_norm(attention_output + layer_input)

            # do double linear projection to enhance the context representation
            with tf.variable_scope('intermediate'):
                intermediate_output = tf.layers.dense(
                    attention_output,
                    intermediate_size,
                    activation=intermediate_act_fn,
                    kernel_initializer=_mh.create_initializer(initializer_range))
            
            with tf.variable_scope('output'):
                layer_output = tf.layers.dense(
                    intermediate_output,
                    hidden_size,
                    kernel_initializer=_mh.create_initializer(initializer_range))
                layer_output = _mh.dropout(layer_output, hidden_dropout_prob)
                layer_output = _mh.layer_norm(layer_output + attention_output)
                prev_output = layer_output
                all_layer_outputs.append(layer_output)
    
    if do_return_all_layers:
        return all_layer_outputs
    else:
        return all_layer_outputs[-1]

def self_attention_layer(from_tensor,
                         to_tensor,
                         attention_mask=None,
                         num_attention_heads=1,
                         size_per_head=512,
                         query_act=None,
                         key_act=None,
                         value_act=None,
                         attention_probs_dropout_prob=0.0,
                         initializer_range=0.02,
                         batch_size=None,
                         from_seq_length=None,
                         to_seq_length=None):
    """Perform self-attention.
    
    Args:
        from_tensor: float Tensor of shape [batch_size, seq_length, width].
        to_tensor: float Tensor of shape [batch_size, seq_length, width].
        attention_mask: (optional) int32 Tensor of shape [batch_size, seq_length, seq_length],
            where 1 indicates the position can be attended and 0 indicates the position cannot be attended.
        num_attention_heads: int. Number of attention heads in the Transformer.
        size_per_head: int. Size of each attention head.
        query_act: (optional) Activation function for the query transformer.
        key_act: (optional) Activation function for the key transformer.
        value_act: (optional) Activation function for the value transformer.
        attention_probs_dropout_prob: (optional) float.
        initializer_range: float.
        batch_size: (optional) int.
        from_seq_length: (optional) int.
        to_seq_length: (optional) int.
    
    Returns:
        float Tensor of shape [batch_size, from_seq_length, width].
    """
    def transpose_for_scores(input_tensor, batch_size, num_attention_heads,
                             seq_length, size_per_head):
        """Change the order of axes. witdh = num_attention_heads * size_per_head.
        
        Args:
            input_tensor: float Tensor of shape [batch_size, seq_length, width].

        Returns:
            float Tensor of shape [batch_size, num_attention_heads, seq_length, size_per_head].
        """
        output_tensor = tf.reshape(input_tensor, [batch_size, seq_length, num_attention_heads, size_per_head])
        output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
        return output_tensor

    # check the rank
    from_shape = _mh.get_shape_list(from_tensor, expected_rank=3)
    to_shape = _mh.get_shape_list(to_tensor, expected_rank=3)
  
    if len(from_shape) != len(to_shape) != 3:
        _error('The rank of `from_tensor` should match the rank of `to_tensor`, and should be 3')
        raise ValueError

    # calculate the query, key, value
    # from_tensor: [batch_size, seq_length, width] -> query_layer: [batch_size, seq_length, num_attention_heads * size_per_head]
    # num_attention_heads * size_per_head == hidden_size == width
    query_layer = tf.layers.dense(from_tensor, 
                                  num_attention_heads * size_per_head,
                                  activation=query_act,
                                  name='query',
                                  kernel_initializer=_mh.create_initializer(initializer_range))

    key_layer = tf.layers.dense(to_tensor,
                                num_attention_heads * size_per_head,
                                activation=key_act,
                                name='key',
                                kernel_initializer=_mh.create_initializer(initializer_range))
  
    value_layer = tf.layers.dense(to_tensor,
                                  num_attention_heads * size_per_head,
                                  activation=value_act,
                                  name='value',
                                  kernel_initializer=_mh.create_initializer(initializer_range))

    # [batch_size, seq_length, width] -> [batch_size, num_attention_heads, seq_length, size_per_head]
    query_layer = transpose_for_scores(query_layer, batch_size,
                                       num_attention_heads, from_seq_length,
                                       size_per_head)
    key_layer = transpose_for_scores(key_layer, batch_size,
                                     num_attention_heads, to_seq_length,
                                     size_per_head)
    
    # calculate the attention scores
    # [batch_size, num_attention_heads, from_seq_length, to_seq_length]
    attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
    attention_scores = tf.multiply(attention_scores, 1.0 / math.sqrt(float(size_per_head)))

    if attention_mask is not None:
        # [batch_size, seq_length, seq_length] -> [batch_size, 1, seq_length, seq_length]
        attention_mask = tf.expand_dims(attention_mask, axis=1)
        adder = (1.0 - tf.cast(attention_mask, dtype=tf.float32)) * -10000.0
        attention_scores += adder
    
    attention_probs = tf.nn.softmax(attention_scores)
    attention_probs = _mh.dropout(attention_probs, attention_probs_dropout_prob)

    # calculate the context layer
    # [batch_size, num_attention_heads, to_seq_length, size_per_head]
    value_layer = transpose_for_scores(value_layer, batch_size,
                                       num_attention_heads, to_seq_length,
                                       size_per_head)
    context_layer = tf.matmul(attention_scores, value_layer)
    # [batch_size, from_seq_length, num_attention_heads, size_per_head]
    context_layer = tf.transpose(context_layer, [0, 2, 1, 3])
    # [batch_size, from_seq_length, width]
    context_layer = tf.reshape(context_layer, [batch_size, from_seq_length, num_attention_heads * size_per_head])

    return context_layer