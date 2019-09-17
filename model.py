# coding:utf-8
# Produced by Andysin Zhang
# 17_Sep_2019
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

import copy
import collections
import numpy as np
import tensorflow as tf
import model_helper as _mh

from log import log_info as _info
from log import log_error as _error

config = collections.namedtuple('Config', 'hidden_dropout_prob attention_prob_dropout_prob')

class BertModel(object):
    def __init__(self, config, is_training, scope=None):
        config = copy.deepcopy(config)
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.attention_prob_dropout_prob = 0.0

        # Initializer Section
        # set the global initializer, which would cover all the variable scopes
        self.initializer = _mh.select_initializer(itype=config.initializer, seed=config.seed, init_weight=config.init_weight)
        tf.get_variable_scope().set_initializer(self.initializer)

        # Input Section
        self.input_ids = tf.placeholder(tf.float32, [None, None], name='input_ids')
        batch_size= self.input_ids.shape.as_list()[0]
        seq_length = self.input_ids.as_list()[1]

        self.input_mask = tf.placeholder(tf.int32, [None, None], name='input_mask')

        # Encoder Section
        # TODO access the variables from the specific scope
        with tf.variable_scope(scope, default_name='bert'):
            # create embedding and get embedded input
            with tf.variable_scope('embeddings'):
                self.embedding = tf.get_variable('embedding', [config.vocab_size, config.embedding_size], dtype=tf.float32)
                embedded_input = tf.nn.embedding_loopup(self.embedding, self.input_ids)
            # add positional embedding
            embedded_input_pos = self._embedding_positional(config.pos_type, embedded_input, dropout_prob=config.dropout_prob)

            # Encoder Blocks
            with tf.variable_scope('encoder'):
                # get attention mask
                attention_mask = self._create_attention_mask(self.input_ids, self.input_mask)
                # Multi-head, multi-layer Transformer
                self.sequence_output = self.transformer_model(embedded_input_pos,
                                                              config.encoder_layer,
                                                              config.num_attention_heads,
                                                              config.forward_size,
                                                              config.forward_ac,
                                                              config.hidden_dropout_prob,
                                                              config.attention_prob_dropout_prob,
                                                              attention_mask,
                                                              config.attention_ac)
            # TODO Decoder
            # Inherit this class, for expansion capability
            # sequence_output: [batch_size, seq_length, embedding_size]
            # use mask to predict the masked words

    def _embedding_positional(self, pos_type, embedded_input, dropout_prob, name=None, max_position_embedding=100):
        """add positional embeddings to the original embeddings.

        Args:
            pos_type: the positional type to use, either 'normal' or 'positional_embedding'.
            embedded_input: original embeddings, [batch_size, seq_length, embedding_size].
            dropout_prob: dropout probability, refer to the 'rate' parameter in tf.nn.dropout()
            max_position_embedding: for 'normal' type, the model learn a new positional matrix,
                so set a max sequence length.
        
        Returns:
            output: identical type and shape to the embedded input.
        """
        input_shape = embedded_input.shape.as_list()
        seq_length, embeded_size = input_shape[1], input_shape[2]

        assert_op = tf.assert_less_equal(seq_length, max_position_embedding)
        with tf.control_dependencies([assert_op]):
            # select sin & cos or normal positional embedding
            if pos_type == 'normal':
                positional_embeddings = tf.get_variable(
                    name='positional_embedding',
                    shape=[max_position_embedding, embeded_size],
                    dtype=tf.float32)
                # slice the positional embeddings according to the actual length
                ac_pos_embed = tf.slice(positional_embeddings, [0, 0], [seq_length, -1])
                embedded_input += ac_pos_embed
            elif pos_type == 'trigonometrical':
                positional_embeddings = np.array(
                    [[pos / np.power(10000, (j - j%2)/embeded_size) for j in range(embeded_size)]
                    for pos in range(seq_length)])
                positional_embeddings[:, 0::2] = np.sin(positional_embeddings[:, 0::2])
                positional_embeddings[:, 1::2] = np.cos(positional_embeddings[:, 1::2])
                positional_embeddings = tf.convert_to_tensor(positional_embeddings)
                embedded_input += positional_embeddings
            else:
                _error('unknown positional type <{}>'.format(pos_type), head='ERROR')
                raise ValueError

        output = self._layer_norm_and_dropout(embedded_input, dropout_prob, name)
        return output

    def _layer_norm_and_dropout(self, input_tensor, dropout_prob, name=None):
        output_tensor = self._layer_norm(input_tensor, name)
        output_tensor = self._dropout(output_tensor, dropout_prob) 
        return output_tensor

    def _layer_norm(self, input_tensor, name=None):
        return tf.contrib.layers._layer_norm(inputs=input_tensor, scope=name)
    
    def _dropout(self, input_tensor, dropout_prob):
        if dropout_prob is None or dropout_prob == 0.0:
            return input_tensor
        return tf.nn.dropout(input_tensor, rate=dropout_prob)

    def _create_attention_mask(self, input_idx, input_mask):
        """create mask for attention matrix.

        Args:
            input_idx: [batch_size, seq_length].
            input_maks: [batch_size, seq_length].
        
        Returns:
            mask: [batch_size, seq_length, seq_length].
        """
        input_shape = input_idx.shape.as_list()
        batch_size, seq_length = input_shape[0], input_shape[1]

        # initial_mask: [b, s, 1]     input_mask: [b, 1, s]
        #     b_1 : 1 1 1 1 1             b_1 : 1 1 0 1 1
        #     b_2 : 1 1 1 1 1             b_2 : 1 0 1 1 1
        #     b_3 : 1 1 1 1 1             b_3 : 1 1 1 1 0
        #
        # mask: [batch_size, seq_length, seq_length]
        #          1 1 0 1 1       1 0 1 1 1       1 1 1 1 0
        #          1 1 0 1 1       1 0 1 1 1       1 1 1 1 0
        #     b_1: 1 1 0 1 1  b_2: 1 0 1 1 1  b_3: 1 1 1 1 0
        #          1 1 0 1 1       1 0 1 1 1       1 1 1 1 0
        #          1 1 0 1 1       1 0 1 1 1       1 1 1 1 0 
        initial_mask = tf.ones([batch_size, seq_length, 1], dtype=tf.float32)    
        input_mask = tf.cast(tf.reshape(input_mask, [batch_size, 1, seq_length]), tf.float32)

        mask = initial_mask * initial_mask
        
        return mask

    def transformer_model(self,
                          input_tensor,
                          encoder_layer,
                          num_attention_heads,
                          forward_size,
                          forward_ac,
                          hidden_dropout_prob,
                          attention_prob_dropout_prob,
                          attention_mask=None,
                          attention_ac=None):
        """Transformer Block.

        Args:
            input_tensor: [batch_size, seq_length, embedding_size].
            encoder_layer: number of transformer blocks.
            num_attention_heads: number of attention heads.
            forward_size: hidden size for the forward layer in a attention block.
        
        Returns:
            all_layer_outputs: a list consists of outputs from attention blocks,
                each output is: [batch_size, seq_length, embedding_size] 
        """
        input_shape = input_tensor.shape.as_list()
        hidden_size = input_shape[2]
        prev_input = input_tensor
        
        assert_op_1 = tf.assert_equal(hidden_size, forward_size)
        with tf.control_dependencies([assert_op_1]):
            all_layer_outputs = []
            for layer_idx in range(encoder_layer):
                layer_input = prev_input
                assert_op_2 = tf.assert_equal(hidden_size % num_attention_heads, 0)
                with tf.control_dependencies([assert_op_2]):
                    attention_head_size = int(hidden_size / num_attention_heads)

                    with tf.variable_scope('layer_{}'.format(layer_idx)):
                        # each block consists of one self-attention and one forward layer
                        with tf.variable_scope('attention_block'):
                            # self-attention
                            with tf.variable_scope('self_attention'):
                                attention_head = self.self_attention(layer_input,
                                                                    num_attention_heads,
                                                                    attention_head_size,
                                                                    attention_prob_dropout_prob,
                                                                    attention_mask,
                                                                    attention_ac)
                            attention_head = self._layer_norm(attention_head + layer_input)

                            # forward layer
                            with tf.variable_scope('inter_forward'):
                                layer_output = tf.layers.dense(
                                    attention_head,
                                    forward_size,
                                    activation=forward_ac,
                                    name='inter_forward')
                            prev_input = self._layer_norm(attention_head + layer_output)
                            all_layer_outputs.append(prev_input)
        
        return all_layer_outputs[-1]

    def self_attention(self,
                       input_tensor,
                       num_attention_heads,
                       attention_head_size,
                       attention_prob_dropout_prob,
                       attention_mask=None,
                       attention_ac=None):
        """multi-headed attention.
        
        Args:
            input_tensor: [batch_size, seq_length, embedding_size].
            num_attention_heads: number of the heads.
            attention_head_size: dimension for each head,
                generally, embedding_size = num_attention_heads * attention_head_size,
                in order to reduce the number of variables.
            attention_prob_dropout_prob: dropout probability.
            attention_mask: mask for maksed words.
            attention_ac: activation function.
        
        Returns:
            context_layer: [batch_size, seq_length, embedding_size]
        """
        # B: batch_size
        # S: seq_length
        # H: hidden_size
        # N: num_attention_heads
        # h: hidden_size for each attention head

        # [B, S, E]
        input_shape = input_tensor.shape.as_list()
        batch_size, seq_length = input_shape[0], input_shape[1]
        
        # [B, S, h * N]
        query_layer = tf.layers.dense(
            input_tensor,
            num_attention_heads * attention_head_size,
            activation=attention_ac,
            name='query')
        
        # [B, S, h * N]
        key_layer = tf.layers.dense(
            input_tensor,
            num_attention_heads * attention_head_size,
            activation=attention_ac,
            name='key')
        
        # [B, S, h * N]
        value_layer = tf.layers.dense(
            input_tensor,
            num_attention_heads * attention_head_size,
            activation=attention_ac,
            name='value')

        # reshape the query and the key,
        # because split the hidden to the num_attention_heads parts,
        # each batch, i.e. seq_length has num_attention_heads queries, keys, values,
        # the num_attention_head axis should be ahead of seq_length axis
        # query: [B, N, S, h] key: [B, N, h, S]
        query_layer = tf.transpose(
                tf.shape(query_layer, [batch_size, seq_length, num_attention_heads, attention_head_size]),
                [0, 2, 1, 3])
        key_layer = tf.transpose(
                tf.reshape(key_layer, [batch_size, seq_length, num_attention_heads, attention_head_size]),
                [0, 2, 3, 1])
        
        # attention_scores: [B, N, S, S]
        attention_scores = tf.matmul(query_layer, key_layer)
        attention_scores = tf.multiply(attention_scores,
                                       1.0 / np.sqrt(float(attention_head_size)))
        if attention_mask is not None:
            # attention mask should be [1, 1, 1, 0, 1] where 1 refers to the non-masked word
            attention_mask_ = (1.0 - attention_mask) * -10000.0
            # [B, 1, S, S]
            attention_mask_ = tf.expand_dims(attention_mask_, axis=[1])
            attention_scores += attention_mask_
        attention_scores = tf.nn.softmax(attention_scores)
        attention_scores = self._dropout(attention_scores, attention_prob_dropout_prob)

        value_layer = tf.transpose(
            tf.reshape(value_layer, [batch_size, seq_length, num_attention_heads, attention_head_size]),
            [0, 2, 1, 3])
        # [B, N, S, h]
        context_layer = tf.matmul(attention_scores, value_layer)
        # [B, S, N, h]
        context_layer = tf.transpose(context_layer, [0, 2, 1, 3])
        # [B, S, N * h]
        context_layer = tf.reshape(context_layer, [batch_size, seq_length, num_attention_heads * attention_head_size])

        return context_layer

if __name__ == '__main__':
    config.hidden_dropout_prob = 0.2 
    config.attention_prob_dropout_prob = 0.2

    print(config.hidden_dropout_prob)