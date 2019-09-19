# coding: utf-8

import numpy as np
import tensorflow as tf
from log import log_info as _info
from log import log_error as _error

__all__ = ['select_initializer', 'get_specific_scope_params', 'create_pos_embeddings']

# Initializer
def select_initializer(itype=None, seed=None, init_weight=0.01):
    if itype.upper() == 'UNIFORM':
        return tf.random_uniform_initializer(-init_weight, init_weight, seed=seed)
    elif itype.upper() == 'GLOROT_N':
        return tf.contrib.keras.initializer.glorot_normal(seed=seed)
    elif itype.upper() == 'GLOROT_U':
        return tf.contrib.keras.initializer.glorot_uniform(seed=seed)
    elif itype.upper() == 'RANDOM':
        return tf.random_normal_initializer(mean=0.0, stddev=init_weight, seed=seed, dtype=tf.float32)
    else:
        _error('Not support <{}> initializer'.format(itype), head='ERROR')
        raise ValueError

# Obtain parameters
def get_specific_scope_params(scope=''):
    """return variables belonging to the specific scope"""
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

# Model
def create_or_load(model, ckpt_path, session, force=False):
    """create a new model or load from the existing one"""
    dir_path = '/'.join(ckpt_path.split('/')[:-1])
    latest_ckpt = tf.train.latest_checkpoint(dir_path)
    
    if latest_ckpt and not force:
        try:
            model.saver.restore(session, latest_ckpt)
        except Exception as e:
            _error(e, head='ERROR')
            raise e
        _info('successfully load model from <{}>'.format(latest_ckpt), head='INFO')
    else:
        session.run(tf.global_variables_initializer())
        session.run(tf.local_variables_initializer())
        session.run(tf.tables_initializer())
        _info('successfully create a new model', head='INFO')
    global_step = model.global_step.eval(session=session)
    return model,global_step

# Positional Embeddings
def create_pos_embeddings(embeded_size, input_length):
    """due to the limitations of the static graph,
       need to create positional embeddings outside.
    """
    positional_embeddings = np.array(
        [[pos / np.power(10000, (j - j%2)/embeded_size) for j in range(embeded_size)]
        for pos in range(input_length)])

    positional_embeddings[:, 0::2] = np.sin(positional_embeddings[:, 0::2])
    positional_embeddings[:, 1::2] = np.cos(positional_embeddings[:, 1::2])
    
    return positional_embeddings