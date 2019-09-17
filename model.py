import copy
import collections
import tensorflow as tf
import model_helper as _mh

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
            embedded_input_pos = self._embedding_positional()

    def _embedding_positional(self, embedded_input):
        input_shape = embedded_input.shape.as_list()
        batch_size, seq_length, embeded_size = input_shape[0], input_shape[1], input_shape[2]

        # TODO select sin & cos or normal positional embedding



if __name__ == '__main__':
    config.hidden_dropout_prob = 0.2 
    config.attention_prob_dropout_prob = 0.2

    print(config.hidden_dropout_prob)