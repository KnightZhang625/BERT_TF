import copy
import collections
import tensorflow as tf


config = collections.namedtuple('Config', 'hidden_dropout_prob attention_prob_dropout_prob')

class BertModel(object):
    def __init__(self, config, is_training, input_ids, input_mask, scope=None):
        config = copy.deepcopy(config)
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.attention_prob_dropout_prob = 0.0
        
        input_ids = tf.placeholder(tf.float32, [None, None], name='input_ids')
        batch_size= input_ids.shape.as_list()[0]
        seq_length = input_ids.as_list()[1]

        if input_mask is None:
            input_mask = tf.ones(shape=[batch_size, seq_length])

        




if __name__ == '__main__':
    config.hidden_dropout_prob = 0.2 
    config.attention_prob_dropout_prob = 0.2

    print(config.hidden_dropout_prob)