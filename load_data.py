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

with codecs.open('data/vocab.txt') as file:
    vocab_idx = {}
    idx_vocab = {}
    for idx, vocab in enumerate(file):
        vocab = vocab.strip()
        idx = int(idx)
        vocab_idx[vocab] = idx
        idx_vocab[idx] = vocab

with codecs.open('data/vocab_idx.pt', 'rb') as file, \
     codecs.open('data/idx_vocab.pt', 'rb') as file_2:
    # pickle.dump(vocab_idx, file, protocol=2)
    # pickle.dump(idx_vocab, file_2, protocol=2)
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
                # add start flag (<s>) and end flag (<\s>) to both question and answer
                que = [vocab_idx['<s>']] + que + [vocab_idx['<\s>']]
                ans  = [vocab_idx['<s>']] + ans + [vocab_idx['<\s>']]
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
    
def train_generator(path, max_length, train_type=None):
    """"This is the entrance to the input_fn."""
    if train_type == 'seq2seq':
        questions, answers, max_length = parse_data(path, train_type)
        for que, ans in zip(questions, answers):
            # 1. input_ids
            # use <mask> to represent the answer instead of the original 0
            input_ids = que + [vocab_idx['<mask>'] for _ in range(len(ans))]    # que + ans(represented by <mask>)
            padding_part = [vocab_idx['<padding>'] for _ in range(max_length - len(input_ids))]
            # input_ids -> [5, 2, 1, 10, 10, 10, 0, 0, 0, 0], where supposing 10 is <mask>, 0 is <padding>
            input_ids += padding_part   # [max_length]

            # 2. mask for attention scores
            # original input_mask in paper -> [1, 1, 1, 0, 0], however, use another mask here
            # where 1 indicates the question part, 0 indicates both the answer part and padding part.
            input_mask = [1 for _ in range(len(que))] + [0 for _ in range(len(ans + padding_part))]
            input_mask = create_mask_for_seq(input_mask, len(que), len(ans + padding_part))
         
            # 3. masked_lm_positions saves the relative positions for answer part and padding part.
            # no padding masked_lm_positions -> [[2, 3, 4, 5, 6, 7, 8, 9], [5, 6, 7, 8, 9]]
            masked_lm_positions = [len(que) + idx for idx in range(max_length - len(que))]
            # ATTENTION # the above `masked_lm_positions` of each data in a batch may not have the same length,
            # # # # # # # due to the various length of question,
            # # # # # # # so padding the `masked_lm_positions` to the same length as max_length is necessary,
            # # # # # # # although the padding items are fake, the following `mask_lm_weights` will handle this.
            # supposing the max_length equals to 10, the example no padding masked_lm_positions will look like
            # the following after the next step:
            # [[2, 3, 4, 5, 6, 7, 8, 9, 0, 0], [5, 6, 7, 8, 0, 0, 0, 0, 0, 0]]
            # The reason for using `0` to pad here, During training, the `masked_lm_positions` wii add the `flat_offset`,
            # the padding items do not exist, if add other numbers instead of 0, maybe cause index error.
            masked_lm_positions += [0 for idx in range(max_length - len(masked_lm_positions))]

            # 4. mask_lm_ids -> the actual labels
            mask_lm_ids = ans + padding_part
            # padding the `mask_lm_ids` to the max_length
            mask_lm_ids += [vocab_idx['<padding>'] for _ in range(max_length - len(mask_lm_ids))]

            # 5. mask_lm_weights -> for calculate the actual loss, which help to ignore the padding part
            mask_lm_weights = [1 for _ in range(len(ans))] + [0 for _ in range(len(padding_part))]
            # padding
            mask_lm_weights += [0 for _ in range(max_length - len(mask_lm_weights))]
            
            # print(input_ids)
            # print(input_mask)
            # print(masked_lm_positions)
            # print(mask_lm_ids)
            # print(mask_lm_weights)
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

def train_input_fn(path, batch_size, repeat_num, max_length=30, train_type=None):
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
        functools.partial(train_generator, path, max_length, train_type),
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
    for i in train_generator('data/train.data', max_length=20, train_type='seq2seq'):
        print(i)