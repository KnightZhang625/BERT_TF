# coding:utf-8

import tensorflow as tf
from log import log_error as _error

__all__ = ['config']

def forbid_new_attributes(wrapped_setatrr):
    def __setattr__(self, name, value):
        if hasattr(self, name):
            wrapped_setatrr(self, name, value)
        else:
            _error('Add new {} is forbidden'.format(name))
            raise AttributeError
    return __setattr__

class NoNewAttrs(object):
    """forbid to add new attributes"""
    __setattr__ = forbid_new_attributes(object.__setattr__)
    class __metaclass__(type):
        __setattr__ = forbid_new_attributes(type.__setattr__)

class Config(NoNewAttrs):
    # dropout
    dropout_prob = 0.2
    hidden_dropout_prob = 0.2
    attention_prob_dropout_prob = 0.2
    
    # initializer
    initializer = 'uniform'
    seed = None
    init_weight = 0.01

    # dimension
    vocab_size = 200
    embedding_size = 32

    # positional embedding type
    # if use trigonometrical, please feed positional embeddings to the model,
    # and vice versa for using normal
    pos_type = 'trigonometrical'

    # Transformer
    encoder_layer = 4
    num_attention_heads = 4
    forward_size = 32
    forward_ac = 'relu'
    attention_ac = None

    # update
    learning_rate =0.1
    decay_step = 1000
    lr_limit = 1e-4

    # save model path
    ckpt_path = 'models/bert-ckpt'
    summary_name = 'tensorboard'

    # train
    steps = 10
    batch_size = 2
    train_type = 0  # 0 for seq2seq, 1 for left-to-right, this parameter is just for UniLM

config = Config()