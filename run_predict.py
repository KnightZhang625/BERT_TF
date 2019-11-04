# coding:utf-8
# Restore model from pb file and do prediction

import sys
import codecs
import pickle
import numpy as np
from tensorflow.contrib import predictor

from pathlib import Path
PROJECT_PATH = Path(__file__).absolute().parent
sys.path.insert(0, str(PROJECT_PATH))

from utils.log import log_error as _error
from load_data import convert_to_idx, create_mask_for_lm, create_mask_for_seq

with codecs.open('data/vocab_idx.pt', 'rb') as file, \
     codecs.open('data/idx_vocab.pt', 'rb') as file_2:
    vocab_idx = pickle.load(file)
    idx_vocab = pickle.load(file_2)

class bertPredict(object):
    def __init__(self, pb_path):
        subdirs = [x for x in Path(pb_path).iterdir()
                   if x.is_dir() and 'temp' not in str(x)]
        latest = str(sorted(subdirs)[-1])

        self.predict_fn = predictor.from_saved_model(latest)
        self.vocab_idx, self.idx_vocab = vocab_idx, idx_vocab
        
    def predict(self, input_ids, max_length):
        input_ids = convert_to_idx(input_ids)
    
        input_ids, input_mask, masked_lm_positions = self._process_input(input_ids, max_length)

        input_ids = np.array(input_ids, dtype=np.int32)
        input_mask = np.array(input_mask, dtype=np.int32)
        masked_lm_positions = np.array(masked_lm_positions, dtype=np.int32)

        # print(input_ids)
        # print(input_mask)
        # print(masked_lm_positions)
        # input()
        result = self.predict_fn(
            {'input_ids': input_ids,
             'input_mask': input_mask,
             'masked_lm_positions': masked_lm_positions})
        return result

    def _process_input(self, input_ids, max_length):
        assert len(input_ids) < max_length, _error('Input length is larger than the maximum length')

        question_length = len(input_ids)
        input_ids += [vocab_idx['<mask>'] for _ in range(max_length - question_length)]
        # input_ids[2] = 330
        # input_ids[3] = 1470
        # input_ids[4] = 1048
        # input_ids[5] = 116
        input_mask = [1 for _ in range(question_length)] + [0 for _ in range(max_length - question_length)]
        input_mask = create_mask_for_seq(input_mask, question_length, max_length - question_length)
        masked_lm_positions = [question_length + idx for idx in range(max_length - question_length)]

        return [input_ids], [input_mask], [masked_lm_positions]

    # def _load_vocab(self, vocab_path):
    #     with codecs.open(vocab_path, 'r', 'utf-8') as file:
    #         vocab_idx = {}
    #         idx_vocab = {}
    #         for idx, vocab in enumerate(file):
    #             vocab = vocab.strip()
    #             vocab_idx[vocab] = idx
    #             idx_vocab[idx] = vocab
    #     print('r', vocab_idx['你'])
    #     print('r', idx_vocab[871])
    #     return vocab_idx, idx_vocab

if __name__ == '__main__':
    bert = bertPredict('models_to_deploy')
    test_tensence = '<s> 你 好 <\s>'
    result = bert.predict(test_tensence, max_length=30)

    # c = 0
    # while (result['output'][c] != bert.vocab_idx['[SEP]']) and (c <= 30):
    #     test_tensence = test_tensence + ' ' + idx_vocab[result['output'][0]]
    #     print(result)

    #     result = bert.predict(test_tensence, max_length=30)
    #     c += 1
    #     print(test_tensence)

    # result = bert.predict(test_tensence, max_length=30)
    for idx in result['output']:
        if idx == bert.vocab_idx['<\s>']:
            break
        else:
            print(bert.idx_vocab[idx])
    