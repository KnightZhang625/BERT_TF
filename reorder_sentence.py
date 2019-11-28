# coding:utf-8

import sys
import copy
import random
import codecs
import pickle
import functools
from pathlib import Path
PROJECT_PATH = Path(__file__).absolute().parent
sys.path.insert(0, str(PROJECT_PATH))

from utils.log import log_info as _info
from utils.log import log_error as _error

with codecs.open('data/vocab_idx_new.pt', 'rb') as file, \
     codecs.open('data/idx_vocab_new.pt', 'rb') as file_2:
    vocab_idx = pickle.load(file)
    idx_vocab = pickle.load(file_2)

__all__ = ['reorder']

def preprocess(func):
    @functools.wraps(func)
    def preprocess_inner(line, ratio=None):
        line_copy = copy.deepcopy(line)
        length = len(line_copy)

        if ratio is None:
            ratio = random.choice([0.5, 0.6, 0.7])
        length_change = int(length * ratio)

        # choose change or remove index
        index_all = list(range(length))
        random.shuffle(index_all)
        index_candidates = [index_all.pop() for _ in range(length_change)]

        return func(line_copy, index_candidates)
    
    return preprocess_inner

@preprocess
def replace_char(line_copy, index_candidates):
    """Replace the chars in the line.
    
    Args:
        line: a list contains the vocabs.
        ratio: the percentage of the vocabs in the line to be replaced.
                Default None, replace chat with uncertain ratios.
    
    Returns:
        a list.
    """        
    for idx in index_candidates:
        line_copy[idx] = idx_vocab[random.choice(range(100, 10000))]

    return line_copy

@preprocess
def cut_line(line_copy, index_candidates):
    """Cut the sentence.
    
    Args:
        line: a list contains the vocabs.
        ratio: the percentage of the parts of the line to be removed.
                Default None, ramove randomly.
    
    Returns:
        a list.
    """
    length = len(line_copy)
    return [line_copy[index] for index in range(length) if index not in index_candidates]

def reorder(sentence):
    """This function is used for reordering the sentence.
    
    Args:
        sentence: str, original sentence.

    Returns:
        reordered sentence as str type.
    """
    line = [v for v in sentence.strip()]
    length = len(line)

    # for length less than or equal to 3, reorder most of the sentence
    if length <= 3:
        new_line = replace_char(line)
    else:
        func = random.choice([replace_char, cut_line])
        new_line = func(line)
    
    return ''.join(new_line)

if __name__ == '__main__':
    print(reorder('我不知道啊'))