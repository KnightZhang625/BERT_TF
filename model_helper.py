import tensorflow as tf
from log import log_info as _info
from log import log_error as _error

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
