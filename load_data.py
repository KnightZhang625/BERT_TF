# coding:utf-8
# load data to the model

import sys
import codecs
import functools
import tensorflow as tf

from pathlib import Path
PROJECT_PATH = Path(__file__).absolute().parent
sys.path.insert(0, str(PROJECT_PATH))

__name__ == ['train_input_fn', 'serving_input_receiver_fn', 'convert_to_idx']

with codecs.open('data/vocab.data') as file:
    vocab_idx = {}
    idx_vocab = {}
    for idx, vocab in enumerate(file):
        vocab = vocab.strip()
        idx = int(idx)
        vocab_idx[vocab] = idx
        idx_vocab[idx] = vocab
    
def convert_to_idx(line):
    """convert the vocab to idx."""
    result = []
    for vocab in line:
        try:
            result.append(vocab_idx[vocab])
        except KeyError:
            result.append(vocab_idx['<unk>'])
    
    return result

def parse_data(path):
    """process the data."""
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

def train_generator(path):
    """"This is the entrance to the input_fn."""
    questions, answers, max_length = parse_data(path)
    for que, ans in zip(questions, answers):
        input_ids = que + ans
        padding_part = [vocab_idx['<padding>'] for _ in range(max_length - len(input_ids))]
        input_ids += padding_part

        # input_mask: -> [1, 1, 1, 0, 0],
        # where 1 indicates the question part, 0 indicates both the answer part and padding part.
        input_mask = [1 for _ in range(len(que))] + [0 for _ in range(len(ans + padding_part))]
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

        # input_ids = [input_ids]
        # input_mask = [input_mask]
        # masked_lm_positions = [masked_lm_positions]
        # mask_lm_ids = [mask_lm_ids]
        # mask_lm_weights = [mask_lm_weights]

        # print(que)
        # print(ans)
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

def train_input_fn(path, batch_size, repeat_num):
    output_types = {'input_ids': tf.int32,
                    'input_mask': tf.int32,
                    'masked_lm_positions': tf.int32,
                    'masked_lm_ids': tf.int32,
                    'masked_lm_weights': tf.int32}
    output_shape = {'input_ids': [None],
                    'input_mask': [None],
                    'masked_lm_positions': [None],
                    'masked_lm_ids': [None],
                    'masked_lm_weights': [None]}
    
    dataset = tf.data.Dataset.from_generator(
        functools.partial(train_generator, path),
        output_types=output_types,
        output_shapes=output_shape)
    dataset = dataset.batch(batch_size).repeat(repeat_num)

    return dataset

def serving_input_receiver_fn():
    """For prediction input."""
    input_ids = tf.placeholder(dtype=tf.int32, shape=[1, None], name='input_ids')
    input_mask = tf.placeholder(dtype=tf.int32, shape=[1, None], name='input_mask')
    masked_lm_positions = tf.placeholder(dtype=tf.int32, shape=[1, None], name='masked_lm_postions')

    receiver_tensors = {'input_ids': input_ids, 'input_mask': input_mask, 'masked_lm_positions': masked_lm_positions}
    features = {'input_ids': input_ids,
                'input_mask': input_mask,
                'masked_lm_positions': masked_lm_positions}
                # 'masked_lm_ids': tf.zeros([1, 10], dtype=tf.int32),
                # 'masked_lm_weights': tf.zeros([1, 10], dtype=tf.int32)}

    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

if __name__ == '__main__':
    for i in train_generator('data/train.data'):
        print(i)