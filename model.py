import copy
import collections
import numpy as np
import tensorflow as tf
import model_helper as _mh

from log import _log_info as _info
from log import _log_error as _error

config = collections.namedtuple('Config', 'hidden_dropout_prob attention_prob_dropout_prob')

class BertModel(object):
    def __init__(self, config, is_training, input_ids, input_mask, scope=None):
        config = copy.deepcopy(config)
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.attention_prob_dropout_prob = 0.0

        # Initializer section
        # set the global initializer, which would cover all the variable scopes
        self.initializer = _mh.select_initializer(itype=config.initializer, seed=config.seed, init_weight=config.init_weight)
        tf.get_variable_scope().set_initializer(self.initializer)

        # Input section
        self.input_ids = tf.placeholder(tf.float32, [None, None], name='input_ids')
        batch_size= input_ids.shape.as_list()[0]
        seq_length = input_ids.as_list()[1]

        if input_mask is None:
            input_mask = tf.ones(shape=[batch_size, seq_length])

        # Encoder Section
        # TODO access the variables from the specific scope
        with tf.variable_scope(scope, default_name='bert'):
            # create embedding and get embedded input
            with tf.variable_scope('embeddings'):
                self.embedding = tf.get_variable('embedding', [config.vocab_size, config.embedding_size], dtype=tf.float32)
                embedded_input = tf.nn.embedding_loopup(self.embedding, self.input_ids)
            # add positional embedding
            embedded_input_pos = self._embedding_positional(config.pos_type, embedded_input, dropout_prob=config.dropout_prob)

    def _embedding_positional(self, pos_type, embedded_input, dropout_prob, name=None, max_position_embedding=100):
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

    def _layer_norm(self, input_tensor, name):
        return tf.contrib.layers._layer_norm(inputs=input_tensor, scope=name)
    
    def _dropout(self, input_tensor, dropout_prob):
        if dropout_prob is None or dropout_prob == 0.0:
            return input_tensor
        return tf.nn.dropout(input_tensor, 1.0 - dropout_prob)

if __name__ == '__main__':
    config.hidden_dropout_prob = 0.2 
    config.attention_prob_dropout_prob = 0.2

    print(config.hidden_dropout_prob)