# coding:utf-8

import codecs

from log import log_info as _info
from log import log_error as _error

def analyse(postive_result, negative_result):
    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0

    with codecs.open(postive_result, 'r', 'utf-8') as file:
        for line in file:
            tag = line.split(' ')[1].strip()
            if tag == '1':
                true_positive += 1
            else:
                false_negative += 1
    
    with codecs.open(negative_result, 'r', 'utf-8') as file:
        for line in file:
            tag = line.split(' ')[1].strip()
            if tag == '0':
                true_negative += 1
            else:
                false_positive += 1
    
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1_score = (2 * precision * recall) / (precision + recall)

    _info('', head='The Result')
    print('\t TP: {}\t|  FP: {}\n \t FN: {}\t|  TN: {}\n  Precision: {:.2}\tRecall: {:.2}\n\t  F1_Score: {:.2}'.format(
        true_positive, false_positive, false_negative, true_negative, precision, recall, f1_score))

if __name__ == '__main__':
    analyse('positive', 'negative')