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
import jieba.posseg as pseg
#tf.enable_eager_execution()

from pathlib import Path
PROJECT_PATH = Path(__file__).absolute().parent
sys.path.insert(0, str(PROJECT_PATH))

from config import bert_config 
from reorder_sentence import reorder
from utils.log import log_info as _info
from utils.log import log_error as _error

# with codecs.open('data/vocab_idx_new.pt', 'rb') as file, \
#      codecs.open('data/idx_vocab_new.pt', 'rb') as file_2:
#     vocab_idx = pickle.load(file)
#     idx_vocab = pickle.load(file_2)

# pos_idx = {}
# idx_pos = {}
# with codecs.open(PROJECT_PATH / 'data/pos_vocab', 'r', 'utf-8') as file, \
#      codecs.open(PROJECT_PATH / 'data/pos_idx.pt', 'wb') as file_2, \
#      codecs.open(PROJECT_PATH / 'data/idx_pos.pt', 'wb') as file_3:
#   for idx, line in enumerate(file):
#     line = line.strip()
#     pos_idx[line] = idx
#     idx_pos[idx] = line
#   pickle.dump(pos_idx, file_2, protocol=2)
#   pickle.dump(idx_pos, file_3, protocol=2)
  
with codecs.open('data/pos_idx.pt', 'rb') as file, \
     codecs.open('data/idx_pos.pt', 'rb') as file_2:
    pos_idx = pickle.load(file)
    idx_vocab = pickle.load(file_2)

def convert_to_idx(line):
    """convert the vocab to idx."""
    result = []
    for vocab in line:
        if vocab == '*':
            vocab = '<padding>'
        elif vocab == '&':
            vocab = '<cls>'
        try:
            result.append(pos_idx[vocab])
        except KeyError:
            result.append(pos_idx['<unk>'])

    return result

def padding(line, max_length):
    if len(line) > max_length:
        return line[:max_length]
    else:
        line += ['*' for _ in range(max_length - len(line))]
    return line

cache = [1, 6] 
def train_generator(path, max_length):
    with codecs.open(path, 'r', 'utf-8') as file:
        for line in file:
            # read data
            line = ''.join(line.split(' '))
            if len(line) <= 29:
                cache_copy = copy.deepcopy(cache) 
                line = line.strip()
                original_line = line 
                for _ in range(2):
                    tag = cache_copy.pop()
                    if tag <= 1:
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
                        if len(line) > 29:
                            line = original_line
                            label = int(1)
                    elif tag == 7:
                        line = reorder(original_line, 4)
                        label = int(0)
                        if len(line) > 29:
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

                    pos_result = list(map(list, list(pseg.cut(line))))
                    pos_feature = [part[1] for part in pos_result]
                    pos_feature.insert(0, '&')
                    original_length = len(pos_feature)
                    input_mask = [1 for _ in range(original_length)] + [0 for _ in range(max_length - original_length)]

                    line_padded = padding(pos_feature, max_length)
                    data = convert_to_idx(line_padded)
  
                    features = {'input_ids': np.array(data, dtype=np.int32),
                                'input_mask': np.array(input_mask, dtype=np.int32)}
                    yield (features, np.array([label], dtype=np.int32))
            else:
                continue

def train_input_fn(path, batch_size, repeat_num, max_length=bert_config.max_length):
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
    for data in train_generator(PROJECT_PATH / 'data/lm_data/sample.data', 10):
        print(data)
        input()
    
    #input_fn = train_input_fn(PROJECT_PATH / 'data/lm_data/corpora_all.data', 2, 5, 10)
    #for data in input_fn:
    #    print(data)
    #    input()