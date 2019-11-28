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
# tf.enable_eager_execution()

from pathlib import Path
PROJECT_PATH = Path(__file__).absolute().parent
sys.path.insert(0, str(PROJECT_PATH))

from utils.log import log_info as _info
from utils.log import log_error as _error

with codecs.open('data/vocab_idx_new.pt', 'rb') as file, \
     codecs.open('data/idx_vocab_new.pt', 'rb') as file_2:
    vocab_idx = pickle.load(file)
    idx_vocab = pickle.load(file_2)

CCHAR_RATE = 0.50
RCHAR_RATE = 0.80
RWORD_RATE = 0.25
RPART_RATE = 0.25
CPART_RATE = 0.50
types_and_probs = {'change_char': 0.5, 'reorder_char': 0.25, 'reorder_word': 0.0, 'repeat_part': 0.0, 'clip_part': 0.25}
def aug_text(text, prob, types_and_probs):
    '''
    probs: to control the total probability of augment.
    types_and_probs: a dict to specify the specific type probability.
    e.g.: {'change_char': 0.2, 'reorder_char': 0.2, 'reorder_word': 0.2, 'repeat_part': 0.2, 'clip_part': 0.2}
    '''
    if (prob == 0.0):
        return text
    if (prob > 1.0):
        _error('prob should be less than 1.0', head='ERROR')
        raise ValueError
    if (None == types_and_probs or len(types_and_probs) == 0):
        _error('types_and_probs should not be none.', head='ERROR')
        raise ValueError
    tp_values = list(types_and_probs.values())
    if (np.sum(tp_values) != 1.0):
        _error('the sum of types_and_probs not equal to 1.0.', head='ERROR')
        raise ValueError
    
    if (np.random.randint(1, 101) < 100 * prob):
        tp_choice = np.random.choice(['change_char', 'reorder_char', 'reorder_word', 'repeat_part', 'clip_part'], 1, p=tp_values)[0]
        tlen = len(text)
        if (tp_choice == 'change_char'):
            vcount = int((tlen) * CCHAR_RATE)
            if (vcount > 0):
                try:
                    vids = np.random.randint(1, tlen-1, [vcount])
                except ValueError:
                    _error(text)
                    _error(vcount)
                    raise ValueError
                vnew_chars = np.random.randint(21, 7818, [vcount])
                for i in range(vcount):
                    text[vids[i]] = str(vnew_chars[i])
        elif (tp_choice == 'reorder_char'):
            vcount = int((tlen) * RCHAR_RATE)
            if (vcount > 0):
                vids = np.random.randint(1, tlen-1, [vcount])
                for i in range(vcount):
                    exid = np.random.randint(1, tlen-1)
                    text[exid], text[vids[i]] = text[vids[i]], text[exid]
        elif (tp_choice == 'repeat_part'):
            if (tlen > 3 * 2):
                slice = np.random.randint(1, tlen-1, [2])
                slice.sort()
                start, end = slice[0], slice[1]
                if (end > start):
                    text = text[:start] + text[start:end] * 2 + text[end:]
        elif (tp_choice == 'clip_part'):
            if (tlen > 3 * 2):
                vcount = int((tlen) * CPART_RATE)
                if (vcount > 0):
                    text = text[vcount:] if (np.random.randint(1, 3) == 1) else text[:tlen - vcount]
        return ''.join(text)

    return ''.join(text)

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

def reorder(line):
    line_list = [v for v in line]
    if len(line_list) <= 3:
        return ''.join([random.choice(line_list) for _ in range(len(line_list))])
    else:
        change_numbers = int(0.5 * len(line_list))
        chars_ids = [random.choice(range(100, 10000)) for _ in range(change_numbers)]
        for idx in range(len(chars_ids)):
            line_list[idx + 1] = idx_vocab[chars_ids[idx]]
        return ''.join(line_list)

def train_generator(path, max_length):
    with codecs.open(path, 'r', 'utf-8') as file:
        for line in file:
            # read data
            line = line.strip()

            # whether reorder
            reorder_or_not = random.choice([0, 1])

            # create data line            
            if reorder_or_not == 1:
                line = padding(line, max_length)
                data = convert_to_idx(line)
                label = int(1)
            elif reorder_or_not == 0:
                line = reorder(line)
                line = padding(line, max_length)
                data = convert_to_idx(line)
                label = int(0)
            else:
                _error('{} not support.'.format(reorder_or_not))
                raise ValueError
            
            features = {'input_ids': np.array(data, dtype=np.int32)}
            yield (features, np.array([label], dtype=np.int32))

def train_input_fn(path, batch_size, repeat_num, max_length=30):
    output_types = {'input_ids': tf.int32}
    output_shapes = {'input_ids': [None]}

    dataset = tf.data.Dataset.from_generator(
        functools.partial(train_generator, path, max_length),
        output_types=(output_types, tf.int32),
        output_shapes=(output_shapes, [None]))
    
    dataset = dataset.batch(batch_size).repeat(repeat_num)
    return dataset

def server_input_receiver_fn():
    input_ids = tf.placeholder(tf.int32, shape=[None, None],  name='input_ids')

    receiver_tensors = {'input_ids': input_ids}

    features = {'input_ids': input_ids}
    
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

if __name__ == '__main__':
    for data in train_generator(PROJECT_PATH / 'data/lm_data/sample.data', 10):
        print(data)
        input()
    
    # input_fn = train_input_fn(PROJECT_PATH / 'data/lm_data/sample.data', 2, 5, 10)
    # for data in input_fn:
    #     print(data)
    #     input()