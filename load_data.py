# coding:utf-8
# load data to the model

import sys
import copy
import codecs
import pickle
import functools
import numpy as np
import tensorflow as tf

from pathlib import Path
PROJECT_PATH = Path(__file__).absolute().parent
sys.path.insert(0, str(PROJECT_PATH))

from utils.log import log_info as _info
from utils.log import log_error as _error

__name__ == ['train_input_fn', 'serving_input_receiver_fn', 'convert_to_idx', 'create_mask_for_lm']

# with codecs.open('data/vocab.txt') as file:
#     vocab_idx = {}
#     idx_vocab = {}
#     for idx, vocab in enumerate(file):
#         vocab = vocab.strip()
#         idx = int(idx)
#         vocab_idx[vocab] = idx
#         idx_vocab[idx] = vocab

with codecs.open('data/vocab_idx.pt', 'rb') as file, \
     codecs.open('data/idx_vocab.pt', 'rb') as file_2:
    vocab_idx = pickle.load(file)
    idx_vocab = pickle.load(file_2)

def convert_to_idx(line):
    """convert the vocab to idx."""
    result = []
    for vocab in line:
        try:
            result.append(vocab_idx[vocab])
        except KeyError:
            result.append(vocab_idx['[UNK]'])
    
    return result

def parse_data(path, train_type=None):
    """process the data."""
    if train_type == 'seq2seq':
        with codecs.open(path, 'r', 'utf-8') as file:
            questions = []
            answers = []
            for line in file:
                line = line.strip().split('=')
                que, ans = convert_to_idx(line[0]), convert_to_idx(line[1])
                questions.append(que)
                answers.append(ans)
        assert len(questions) == len(answers)
        
        # get max length to pad
        length = [len(ans) + len(que) for ans, que in zip(questions, answers)]
        max_length = max(length)
        return questions, answers, max_length
    
    elif train_type == 'lm':
        with codecs.open(path, 'r', 'utf-8') as file:
            sentences = []
            for line in file:
                line = line.strip()
                line = convert_to_idx(line)
                sentences.append(line)
        length = [len(line) for line in sentences]
        max_length = max(length)
        return sentences, max_length

def create_mask_for_seq(input_mask, len_que, len_ans_pad):
    """create mask for UniLM seq2seq task.
        This function replace the original mask as [1, 1, 0, 0, 0],
        otherwise, it looks like:
            [1, 1, 0, 0, 0]
            [1, 1, 0, 0, 0]
            [1, 1, 1, 0, 0]
            [1, 1, 1, 1, 0]
            [1, 1, 1, 1, 1]
        because for question, all the words could see each other,
        for answers, the words could see the predicted ones."""
    
    lm_mask = []
    for _ in range(len_que):
        temp = copy.deepcopy(input_mask)
        lm_mask.append(temp)
    
    temp = copy.deepcopy(input_mask)
    for idx in range(len_ans_pad):
        tempp = copy.deepcopy(temp)
        tempp[len_que + idx] = 1
        lm_mask.append(tempp)
        temp = copy.deepcopy(tempp)
    
    return np.array(lm_mask)

def create_mask_for_lm(length):
    """create mask for UniLM language model task."""
    mask = []
    for row in range(length):
        row_mask = [1 for col in range(row + 1)] + [0 for col in range(length - row - 1)]
        mask.append(row_mask)
    
    return np.array(mask)
    
def train_generator(path, train_type=None):
    """"This is the entrance to the input_fn."""
    if train_type == 'seq2seq':
        questions, answers, max_length = parse_data(path, train_type)
        for que, ans in zip(questions, answers):
            # input_ids = que + ans     # maybe the input should not show the answer
            input_ids = que + [0 for _ in range(len(ans))]
            padding_part = [vocab_idx['<padding>'] for _ in range(max_length - len(input_ids))]
            input_ids += padding_part

            # input_mask: -> [1, 1, 1, 0, 0],
            # where 1 indicates the question part, 0 indicates both the answer part and padding part.
            input_mask = [1 for _ in range(len(que))] + [0 for _ in range(len(ans + padding_part))]
            input_mask = create_mask_for_seq(input_mask, len(que), len(ans + padding_part))
         
            # masked_lm_positions saves the relative positions for answer part and padding part.
            # [[2, 3, 4, 5, 6], [5, 6]]
            masked_lm_positions = [idx + len(que) for idx in range(len(input_ids) - len(que))]
            # ATTENTION: the above `masked_lm_positions` is not in the same length due to the various length of question,
            # so padding the `masked_lm_positions` to the same length as input_ids,
            # although the padding items are fake, the following `mask_lm_weights` will handle this.
            masked_lm_positions += [masked_lm_positions[-1]  + 1 + idx  for idx in range(len(input_ids) - len(masked_lm_positions))]
            mask_lm_ids = ans + padding_part
            mask_lm_ids += [vocab_idx['<padding>'] for _ in range(len(input_ids) - len(mask_lm_ids))]
            mask_lm_weights = [1 for _ in range(len(ans))] + [0 for _ in range(len(padding_part))]
            mask_lm_weights += [0 for _ in range(len(input_ids) - len(mask_lm_weights))]
            
            # print(input_ids)
            # print(input_mask)
            # print(masked_lm_positions)
            # print(len(mask_lm_ids))
            # print(len(mask_lm_weights))
            # input()
 
            features = {'input_ids': input_ids,
                        'input_mask': input_mask,
                        'masked_lm_positions': masked_lm_positions,
                        'masked_lm_ids': mask_lm_ids,
                        'masked_lm_weights': mask_lm_weights}
            yield features
    elif train_type == 'lm':
        sentences, max_length = parse_data(path, train_type)
        for line in sentences:
            input_ids = [vocab_idx['S']]
            padding_part = [vocab_idx['<padding>'] for _ in range(max_length - len(input_ids))]
            input_ids += padding_part
            
            input_mask = create_mask_for_lm(max_length)

            masked_lm_positions = [idx + 1 for idx in range(len(input_ids) - 1)]
            masked_lm_positions += [masked_lm_positions[-1] + 1 + idx for idx in range(len(input_ids) - len(masked_lm_positions))]
            mask_lm_ids = line + [vocab_idx['<padding>'] for _ in range(len(input_ids) - len(line) - 1)]
            mask_lm_ids += [vocab_idx['<padding>'] for _ in range(len(input_ids) - len(mask_lm_ids))]
            mask_lm_weights = [1 for _ in range(len(line))] + [0 for _ in range(len(input_ids) - len(line) - 1)]
            mask_lm_weights += [0 for _ in range(len(input_ids) - len(mask_lm_weights))]

            # print(line)
            # print(len(input_ids))
            # print(len(input_mask))
            # print(len(masked_lm_positions))
            # print(len(mask_lm_ids))
            # print(len(mask_lm_weights))
            # input()

            features = {'input_ids': input_ids,
                        'input_mask': input_mask,
                        'masked_lm_positions': masked_lm_positions,
                        'masked_lm_ids': mask_lm_ids,
                        'masked_lm_weights': mask_lm_weights}
            yield features
    else:
        _error('Non supported train type: {}'.format(train_type))
        raise ValueError        

def train_input_fn(path, batch_size, repeat_num, train_type=None):
    output_types = {'input_ids': tf.int32,
                    'input_mask': tf.int32,
                    'masked_lm_positions': tf.int32,
                    'masked_lm_ids': tf.int32,
                    'masked_lm_weights': tf.int32}
    output_shape = {'input_ids': [None],
                    'input_mask': [None, None],
                    'masked_lm_positions': [None],
                    'masked_lm_ids': [None],
                    'masked_lm_weights': [None]}
    
    dataset = tf.data.Dataset.from_generator(
        functools.partial(train_generator, path, train_type),
        output_types=output_types,
        output_shapes=output_shape)
    dataset = dataset.batch(batch_size).repeat(repeat_num)

    return dataset

def serving_input_receiver_fn():
    """For prediction input."""
    input_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name='input_ids')
    input_mask = tf.placeholder(dtype=tf.int32, shape=[None, None, None], name='input_mask')
    masked_lm_positions = tf.placeholder(dtype=tf.int32, shape=[None, None], name='masked_lm_postions')

    receiver_tensors = {'input_ids': input_ids, 'input_mask': input_mask, 'masked_lm_positions': masked_lm_positions}
    features = {'input_ids': input_ids,
                'input_mask': input_mask,
                'masked_lm_positions': masked_lm_positions,
                'masked_lm_ids': tf.zeros([1, 10], dtype=tf.int32),
                'masked_lm_weights': tf.zeros([1, 10], dtype=tf.int32)}

    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

if __name__ == '__main__':
    for i in train_generator('data/train.data', 'seq2seq'):
        print(i)