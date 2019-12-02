# coding:utf-8
# Bert Config

import sys
from pathlib import Path
PROJECT_PATH = Path(__file__).absolute().parent
sys.path.insert(0, str(PROJECT_PATH))

from utils.log import log_error as _error

__name__ = ['cnn_config']

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

class CNNConfig(NoNewAttrs):
        # train
        learning_rate = 1e-2
        lr_limit = 1e-2
        l2_reg_lambda = 1e-4

        # model
        vocab_size = 10509
        embedding_size = 100

        # initializer
        initializer_range = 0.02
        dropout_prob = 0.2

        # global
        data_path = 'data/lm_data/sample.data'
        model_dir = 'models_cnn/'
        batch_size = 2
        num_train_steps = 1000
        max_length = 30

        num_classes = 2
        

cnn_config = CNNConfig()