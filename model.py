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

import os
import copy
import numpy as np
import tensorflow as tf
import model_helper as _mh

from hparams_config import config as config_
from log import log_info as _info
from log import log_error as _error

__all__ = ['BertModel']

class BertModel(object):
    def __init__(self, config, is_training, scope=None):
        config = copy.deepcopy(config_)
        self.is_training = is_training
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.attention_prob_dropout_prob = 0.0

        # Initializer Section
        # set the global initializer, which would cover all the variable scopes
        self.initializer = _mh.select_initializer(itype=config.initializer, seed=config.seed, init_weight=config.init_weight)
        tf.get_variable_scope().set_initializer(self.initializer)
        self.global_step = tf.Variable(0, trainable=False)

        # Input Section
        self.input_ids = tf.placeholder(tf.int32, [None, None], name='input_ids')
        self.input_length = tf.placeholder(tf.int32, name='input_length')
        self.input_mask = tf.placeholder(tf.int32, [None, None], name='input_mask')

        # Output Mask
        self.output_ids = tf.placeholder(tf.int32, [None, None], name='output_ids')

        # Encoder Section
        # TODO access the variables from the specific scope
        with tf.variable_scope(scope, default_name='bert'):
            # create embedding and get embedded input
            with tf.variable_scope('embeddings'):
                self.embedding = tf.get_variable('embedding', [config.vocab_size, config.embedding_size], dtype=tf.float32)
                embedded_input = tf.nn.embedding_lookup(self.embedding, self.input_ids)
            # add positional embedding
            embedded_input_pos = self._embedding_positional(config.pos_type, embedded_input, config.embedding_size, dropout_prob=config.dropout_prob)

            # Encoder Blocks
            with tf.variable_scope('encoder'):
                # get attention mask
                attention_mask = self._create_attention_mask(self.input_ids, self.input_mask, config.batch_size)
                # Multi-head, multi-layer Transformer
                sequence_output = self.transformer_model(embedded_input_pos,
                                                         config.batch_size,
                                                         config.encoder_layer,
                                                         config.num_attention_heads,
                                                         config.forward_size,
                                                         config.forward_ac,
                                                         config.hidden_dropout_prob,
                                                         config.attention_prob_dropout_prob,
                                                         attention_mask,
                                                         config.attention_ac)
            
            # Decoder
            # Inherit this class, for expansion capability,
            # No Returns, because of considerring for expansion after pre_train,
            # do not add to much variable in initial function.
            with tf.variable_scope('decoder'):
                self._projection(sequence_output, config.vocab_size)
            
            if is_training:
                self._compute_loss(config.batch_size)
                self._update(config.learning_rate, config.decay_step, config.lr_limit)
                self.train_summary = tf.summary.merge(
                    [tf.summary.scalar('lr', self.learning_rate),
                    tf.summary.scalar('loss', self.loss_bs)])
            else:
                self._infer()
           
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=3)
            _info('Finish Building Graph', head='INFO')

    def _embedding_positional(self, pos_type, embedded_input, embedding_size, dropout_prob, name=None, max_position_embedding=100):
        """add positional embeddings to the original embeddings.

        Args:
            pos_type: the positional type to use, either 'normal' or 'positional_embeddings'.
            embedded_input: original embeddings, [batch_size, seq_length, embedding_size].
            embedding_size: embedding size.
            dropout_prob: dropout probability, refer to the 'rate' parameter in tf.nn.dropout()
            max_position_embedding: for 'normal' type, the model learn a new positional matrix,
                so set a max sequence length.
        
        Returns:
            output: identical type and shape to the embedded input.
        """
        assert_op = tf.assert_less_equal(self.input_length, max_position_embedding)
        self.pos_type = pos_type
        with tf.control_dependencies([assert_op]):
            # select sin & cos or normal positional embedding
            if pos_type == 'normal':
                positional_embeddings = tf.get_variable(
                    name='positional_embeddings',
                    shape=[max_position_embedding, embedding_size],
                    dtype=tf.float32)
                # slice the positional embeddings according to the actual length
                ac_pos_embed = tf.slice(positional_embeddings, [0, 0], [self.input_length, -1])
                embedded_input += ac_pos_embed
            elif pos_type == 'trigonometrical':
                self.positional_embeddings = tf.placeholder(dtype=tf.float32, shape=[None, None], name='positional_embeddings')
                positional_embeddings = tf.convert_to_tensor(self.positional_embeddings)
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
        return tf.contrib.layers.layer_norm(inputs=input_tensor, scope=name)
    
    def _dropout(self, input_tensor, dropout_prob):
        if dropout_prob is None or dropout_prob == 0.0:
            return input_tensor
        return tf.nn.dropout(input_tensor, rate=dropout_prob)

    def _create_attention_mask(self, input_idx, input_mask, batch_size):
        """create mask for attention matrix.

        Args:
            input_idx: [batch_size, seq_length].
            input_mask: [batch_size, seq_length].
        
        Returns:
            mask: [batch_size, seq_length, seq_length].
        """
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
        initial_mask = tf.ones([batch_size, self.input_length, 1], dtype=tf.float32)    
        input_mask = tf.cast(tf.reshape(input_mask, [batch_size, 1, self.input_length]), tf.float32)

        mask = initial_mask * initial_mask
        
        return mask

    def transformer_model(self,
                          input_tensor,
                          batch_size,
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
                                                                     batch_size,
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
                       batch_size,
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
                tf.reshape(query_layer, [batch_size, self.input_length, num_attention_heads, attention_head_size]),
                [0, 2, 1, 3])
        key_layer = tf.transpose(
                tf.reshape(key_layer, [batch_size, self.input_length, num_attention_heads, attention_head_size]),
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
            tf.reshape(value_layer, [batch_size, self.input_length, num_attention_heads, attention_head_size]),
            [0, 2, 1, 3])
        # [B, N, S, h]
        context_layer = tf.matmul(attention_scores, value_layer)
        # [B, S, N, h]
        context_layer = tf.transpose(context_layer, [0, 2, 1, 3])
        # [B, S, N * h]
        context_layer = tf.reshape(context_layer, [batch_size, self.input_length, num_attention_heads * attention_head_size])

        return context_layer
    
    def _projection(self, sequence_output, vocab_size):
        """project the output from the encoder to the vocab size."""
        with tf.variable_scope('output_layer'):
            self.logits = tf.layers.dense(sequence_output, vocab_size, name='output')
    
    def _compute_loss(self, batch_size):
        """compute the cross-entropy loss."""
        # 1 refers to occurring word, 0 refers to masked word in original input_mask
        # abandon the input_mask, do not really learn the data
        # input_mask = tf.cast(self.input_mask, tf.float32)
        # loss_mask = 1 - input_mask
    
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.output_ids, logits=self.logits)
        self.loss_bs = tf.reduce_sum(loss) / batch_size
    
    def _update(self, learning_rate, decay_step, lr_limit):
        """update the parameters."""
        self.learning_rate = tf.maximum(tf.constant(lr_limit),
                                        tf.train.polynomial_decay(learning_rate, self.global_step, 
                                                                  decay_step, power=0.5, cycle=True))    
        optimizer = tf.train.AdamOptimizer(self.learning_rate, name='Adam')
        parameters = tf.trainable_variables()
        gradients = tf.gradients(self.loss_bs, parameters, colocate_gradients_with_ops=True)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        self.upgrade = optimizer.apply_gradients(zip(clipped_gradients, parameters), global_step=self.global_step)

    def _infer(self):
        """get the prediction index in inference."""
        outputs = tf.nn.softmax(self.logits)
        self.predict_idx = tf.argmax(outputs, axis=2)
    
    def train(self, sess, data):
        assert self.is_training
        if self.pos_type is 'trigonometrical':
            feed = {self.input_ids: data.input_ids,
                    self.input_mask: data.input_mask,
                    self.input_length: data.input_length,
                    self.output_ids: data.output_ids,
                    self.positional_embeddings: data.positional_embeddings}
        else:
            feed = {self.input_ids: data.input_ids,
                    self.input_mask: data.input_mask,
                    self.input_length: data.input_length,
                    self.output_ids: data.output_ids}

        return sess.run([self.global_step, 
                         self.learning_rate, 
                         self.loss_bs,
                         self.upgrade,
                         self.logits,
                         self.train_summary], feed_dict=feed)

    def infer(self, sess, data):
        assert not self.is_training

        if self.pos_type is 'trigonometrical':
            feed = {self.input_ids: data.input_ids,
                    self.input_length: data.input_length,
                    self.input_mask: data.input_mask,
                    self.positional_embeddings: data.positional_embeddings}
        else:
            feed = {self.input_ids: data.input_ids,
                    self.input_length: data.input_length,
                    self.input_mask: data.input_mask}

        return sess.run([self.predict_idx], feed_dict=feed)