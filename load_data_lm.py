# coding:utf-8
# load data to the model

import sys
import copy
import codecs
import pickle
import random
import functools
import numpy as np
import tensorflow as tf
#tf.enable_eager_execution()

from pathlib import Path
PROJECT_PATH = Path(__file__).absolute().parent
sys.path.insert(0, str(PROJECT_PATH))

from reorder_sentence import reorder
from utils.log import log_info as _info
from utils.log import log_error as _error

with codecs.open('data/vocab_idx_new.pt', 'rb') as file, \
     codecs.open('data/idx_vocab_new.pt', 'rb') as file_2:
    vocab_idx = pickle.load(file)
    idx_vocab = pickle.load(file_2)

def convert_to_idx(line):
    """convert the vocab to idx."""
    result = []
    for vocab in line:
        if vocab == 'p':
            vocab = '<padding>'
        try:
            result.append(vocab_idx[vocab])
        except KeyError:
            result.append(vocab_idx['<unk>'])

    return result

def padding(line, max_length):
    if len(line) > max_length:
        return line[:max_length]
    else:
        line += 'p' * (max_length - len(line))
    return line

def train_generator(path, max_length):
    with codecs.open(path, 'r', 'utf-8') as file:
        for line in file:
            # read data
            line = ''.join(line.split(' '))
            if len(line) <= 30:
                line = line.strip()
                original_line = line 
                for tag in range(7):
                    if tag <= 3:
                        label = int(1)
                    elif tag == 4:
                        line = reorder(original_line, 1)
                        label = int(0)
                    elif tag == 5:
                        line = reorder(original_line, 2)
                        label = int(0)
                    elif tag == 6:
                        line = reorder(original_line, 3)
                        label = int(0)
                        if len(line) > 30:
                            line = original_line
                            label = int(1)

                    """
                    # whether reorder
                    reorder_or_not = random.choice([0, 1])

                    # create data line            
                    if reorder_or_not == 1:
                        label = int(1)
                    elif reorder_or_not == 0:
                        line = reorder(line)
                        label = int(0)
                    else:
                        _error('{} not support.'.format(reorder_or_not))
                        raise ValueError
                    """

                    original_length = len(line)
                    input_mask = [1 for _ in range(original_length)] + [0 for _ in range(max_length - original_length)]
                    line_padded = padding(line, max_length)
                    data = convert_to_idx(line_padded)

                    features = {'input_ids': np.array(data, dtype=np.int32),
                                'input_mask': np.array(input_mask, dtype=np.int32)}
                    yield (features, np.array([label], dtype=np.int32))
            else:
                continue

def train_input_fn(path, batch_size, repeat_num, max_length=30):
    output_types = {'input_ids': tf.int32, 'input_mask': tf.int32}
    output_shapes = {'input_ids': [None], 'input_mask': [None]}

    dataset = tf.data.Dataset.from_generator(
        functools.partial(train_generator, path, max_length),
        output_types=(output_types, tf.int32),
        output_shapes=(output_shapes, [None]))
    
    dataset = dataset.batch(batch_size).repeat(repeat_num)
    return dataset

def server_input_receiver_fn():
    input_ids = tf.placeholder(tf.int32, shape=[None, None],  name='input_ids')
    # input_mask = tf.placeholder(tf.int32, shape=[None, None], name='input_mask')
    receiver_tensors = {'input_ids': input_ids}

    features = {'input_ids': input_ids}
    
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

if __name__ == '__main__':
    #for data in train_generator(PROJECT_PATH / 'data/lm_data/corpora_all.data', 10):
    #    print(data)
    #    input()
    
    input_fn = train_input_fn(PROJECT_PATH / 'data/lm_data/sample.data', 2, 5, 10)
    for data in input_fn:
        print(data)
        input()
