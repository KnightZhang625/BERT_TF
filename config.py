# coding:utf-8
# Bert Config

import sys
from pathlib import Path
PROJECT_PATH = Path(__file__).absolute().parent
sys.path.insert(0, str(PROJECT_PATH))

from utils.log import log_error as _error

__name__ = ['bert_config']

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

class BertConfig(NoNewAttrs):
        # train
        learning_rate = 1e-3

        # model
        vocab_size = 7819
        embedding_size = 32
        hidden_size = 64
        max_positional_embeddings = 30
        token_type_vocab_size = 0
        pre_positional_embedding_type = 'normal'
        hidden_dropout_prob = 0.1
        attention_probs_dropout_prob = 0.1 
        num_hidden_layers = int(2)
        num_attention_heads = int(2)
        intermediate_size = 32

        # initializer
        initializer_range = 0.01

        # global
        data_path = 'data/train.data'
        model_dir = 'models/'
        init_checkpoint = None
        batch_size = 5
        num_train_steps = 100

bert_config = BertConfig()