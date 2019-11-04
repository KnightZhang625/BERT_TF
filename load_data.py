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
    # pickle.dump(vocab_idx, file, protocol=2)
    # pickle.dump(idx_vocab, file_2, protocol=2)
    vocab_idx = pickle.load(file)
    idx_vocab = pickle.load(file_2)

def convert_to_idx(line):
    """convert the vocab to idx."""
    result = []
    for vocab in line.split(' '):
        try:
            result.append(vocab_idx[vocab])
        except KeyError:
            result.append(vocab_idx['<unk>'])
    
    return result

def parse_data(path, train_type=None):
    """process the data."""
    if train_type == 'seq2seq' or train_type == 'bi':
        with codecs.open(path, 'r', 'utf-8') as file:
            questions = []
            answers = []
            for line in file:
                line = line.strip().split('=')
                que, ans = convert_to_idx(line[0]), convert_to_idx(line[1])
                # add start flag (<s>) and end flag (<\s>) to both question and answer
                # que = que
                # ans  = ans + [vocab_idx['<\s>']]
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

def create_mask_for_lm(length, reverse=False):
    """create mask for UniLM language model task."""
    mask = []
    for row in range(length):
        if not reverse:
            row_mask = [1 for col in range(row + 1)] + [0 for col in range(length - row - 1)]
        else:
            row_mask = [0 for col in range(length - row - 1)] + [1 for col in range(row + 1)]
        mask.append(row_mask)
    
    return np.array(mask)

def create_mask_for_bi(length):
    """create mask for UniLM Bidirectional LM task."""
    mask = []
    for _ in range(length):
        row_mask = [1 for _ in range(length)]
        mask.append(row_mask)

    return np.array(mask)

def generate_mask(train_type, train_ids, percentage=0.20, offset_number=0, reset=False):
    if reset:
        for idx in range(len(train_ids)):
            train_ids[idx] = vocab_idx['<mask>']
        masked_lm_positions = [offset_number + idx for idx in range(len(train_ids))]
        return train_ids, masked_lm_positions
    if train_type == 'seq2seq' or train_type == 'bi':
        return generate_mask('lm', train_ids, percentage=percentage, offset_number=offset_number)
    elif train_type == 'lm':
        num = int(percentage * len(train_ids)) 
        mask_number = num if num != 0 else 2
        mask_postions = [-1]

        temp_number = -1
        for _ in range(mask_number):
            while (temp_number + offset_number) in mask_postions:
                temp_number = random.randint(0, len(train_ids)-1)
            train_ids[temp_number] = vocab_idx['<mask>']
            mask_postions.append(temp_number + offset_number)
        mask_postions.pop(0)
        return train_ids, sorted(mask_postions)

def train_generator(path, max_length, train_type=None, reverse=False):
    """"This is the entrance to the input_fn."""
    if train_type == 'seq2seq' or train_type == 'bi':
        questions, answers, max_length = parse_data(path, train_type)
        for que, ans in zip(questions, answers):
            # 1. input_ids
            # use <mask> to represent the answer instead of the original 0
            # input_ids = que + [vocab_idx['<mask>'] for _ in range(len(ans))]    # que + ans(represented by <mask>)
            que_no_maks = copy.deepcopy(que)
            ans_no_mask = copy.deepcopy(ans)
            que_ans_no_mask = que_no_maks + ans_no_mask
            if train_type == 'seq2seq':
                # que, que_mask = generate_mask(train_type, que)
                que_mask = []
                ans, ans_mask = generate_mask(train_type, ans, percentage=1, offset_number=len(que)) 
                input_ids = que + ans
                padding_part = [vocab_idx['<padding>'] for _ in range(max_length - len(input_ids))]
                # input_ids -> [5, 2, 1, 10, 10, 10, 0, 0, 0, 0], where supposing 10 is <mask>, 0 is <padding>
                input_ids += padding_part   # [max_length]
            else:
                que, que_mask = generate_mask(train_type, que)
                ans, ans_mask = generate_mask(train_type, ans, offset_number=len(que)) 
                input_ids = que + ans
                padding_part = [vocab_idx['<padding>'] for _ in range(max_length - len(input_ids))]
                # input_ids -> [5, 2, 1, 10, 10, 10, 0, 0, 0, 0], where supposing 10 is <mask>, 0 is <padding>
                input_ids += padding_part   # [max_length]
            # 2. mask for attention scores
            # original input_mask in paper -> [1, 1, 1, 0, 0], however, use another mask here
            # where 1 indicates the question part, 0 indicates both the answer part and padding part.
            if train_type == 'seq2seq':
                input_mask = [1 for _ in range(len(que))] + [0 for _ in range(len(ans + padding_part))]
                input_mask = create_mask_for_seq(input_mask, len(que), len(ans + padding_part))
            else:
                input_mask = create_mask_for_bi(max_length)
         
            # 3. masked_lm_positions saves the relative positions for answer part and padding part.
            # no padding masked_lm_positions -> [[2, 3, 4, 5, 6, 7, 8, 9], [5, 6, 7, 8, 9]]
            # masked_lm_positions = [len(que) + idx for idx in range(max_length - len(que))]
            masked_lm_positions = que_mask + ans_mask
            masked_lm_positions_copy = copy.deepcopy(masked_lm_positions)
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
            # mask_lm_ids = ans + padding_part
            mask_lm_ids = [que_ans_no_mask[idx] for idx in masked_lm_positions]
            # padding the `mask_lm_ids` to the max_length
            mask_lm_ids += [vocab_idx['<padding>'] for _ in range(max_length - len(mask_lm_ids))]

            # 5. mask_lm_weights -> for calculate the actual loss, which help to ignore the padding part
            mask_lm_weights = [1 for _ in range(len(masked_lm_positions_copy))] + [0 for _ in range(max_length - len(masked_lm_positions_copy))]
            # padding
            # mask_lm_weights += [0 for _ in range(max_length - len(mask_lm_weights))]
            
            print(len(input_ids))
            print(len(input_mask))
            print(len(masked_lm_positions))
            print(len(mask_lm_ids))
            print(len(mask_lm_weights))
            input()
 
            features = {'input_ids': input_ids,
                        'input_mask': input_mask,
                        'masked_lm_positions': masked_lm_positions,
                        'masked_lm_ids': mask_lm_ids,
                        'masked_lm_weights': mask_lm_weights}
            yield features
    elif train_type == 'lm':
        sentences, max_length = parse_data(path, train_type)
        for line in sentences:
            input_ids = copy.deepcopy(line)
            input_ids, mask_postions = generate_mask('lm', input_ids)

            padding_part = [vocab_idx['<padding>'] for _ in range(max_length - len(input_ids))]
            input_ids += padding_part
            
            input_mask = create_mask_for_lm(max_length, reverse=reverse)

            masked_lm_positions = copy.deepcopy(mask_postions)
            # masked_lm_positions = [idx + 1 for idx in range(len(input_ids) - 1)]
            masked_lm_positions += [0 for idx in range(max_length- len(masked_lm_positions))]
            mask_lm_ids = [line[idx] for idx in mask_postions]
            mask_lm_ids += [vocab_idx['<padding>'] for _ in range(len(input_ids) - len(mask_lm_ids))]
            mask_lm_weights = [1 for _ in range(len(mask_postions))] + [0 for _ in range(max_length - len(mask_postions))]
            # mask_lm_weights += [0 for _ in range(len(input_ids) - len(mask_lm_weights))]

            features = {'input_ids': input_ids,
                        'input_mask': input_mask,
                        'masked_lm_positions': masked_lm_positions,
                        'masked_lm_ids': mask_lm_ids,
                        'masked_lm_weights': mask_lm_weights}
            yield features
    else:
        _error('Non supported train type: {}'.format(train_type))
        raise ValueError        

def train_input_fn(path, batch_size, repeat_num, max_length=30, train_type=None, reverse=False):
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
        functools.partial(train_generator, path, max_length, train_type, reverse),
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