# coding:utf-8
# For Inference

import sys
import codecs
import functools
import numpy as np
from tensorflow.contrib import predictor

from pathlib import Path
PROJECT_PATH = Path(__file__).absolute().parent

sys.path.insert(0, str(PROJECT_PATH))

from config import bert_config
from load_data_lm import convert_to_idx, padding
from utils.log import log_info as _info
from utils.log import log_error as _error

class NmtPredict(object):
    def __init__(self, pb_path):
        subdirs = [x for x in Path(pb_path).iterdir()
                    if x.is_dir() and 'temp' not in str(x)]
        latest_model = str(sorted(subdirs)[-1])

        self.predict_fn = predictor.from_saved_model(latest_model)

    def batch_predict(flag):
        def _process(func):
            @functools.wraps(func)
            def _process_inner(*args, **kwargs):
                # _process_inner called by object, so the args[0] is 'self',
                # the decorated function is called below, so no need to add self in parameters.
                self = args[0]
                if flag:
                    result = []
                    save_or_not = True
                    if len(args) > 2:
                        threshold = args[2]
                    else:
                        threshold = np.inf
                else:
                    save_or_not = False

                for sentence in func(args[1]):
                    original_sentence = sentence
                    # original_length = len(original_sentence.strip())
                    # input_mask = [1 for _ in range(original_length)] + [0 for _ in range(bert_config.max_length - original_length)]

                    sentence = padding(sentence.strip(), bert_config.max_length)
                    input_ids = convert_to_idx(sentence)

                    features = {'input_ids': np.array([input_ids], dtype=np.int32)}

                    predictions = self.predict_fn(features)
                    predict_id = predictions['class'][0]
                
                    if save_or_not:
                        result.append((original_sentence, predict_id,))
                    else:
                        return original_sentence, predict_id
                
                self.write_result(result, threshold)

            return _process_inner
        return _process

    @batch_predict(flag=False)
    def predict(sentence):
        yield sentence

    @batch_predict(flag=True)
    def predict_batch(path):
        with codecs.open(path, 'r', 'utf-8') as file:
            for line in file:
                line = line.strip()
                if len(line) != 0:
                    yield line
    
    @staticmethod
    def _convert_idx_str(sample_id):
        return idx_to_str(sample_id)
    
    @staticmethod
    def write_result(result, threshold, path=PROJECT_PATH / 'data/test_data/positive_result'):
        with codecs.open(path, 'w', 'utf-8') as file:
            for line in result:
                sentence, predict_id = line[0], line[1]
                to_write = sentence + '\t' + str(predict_id) + '\n'
                file.write(to_write)
                file.flush()
        _info('The result has been saved to {}'.format(path))

if __name__ == '__main__':
    nmt_predict = NmtPredict(PROJECT_PATH / 'models_deploy_lm')
    predict_sentence, predict_id = nmt_predict.predict('圆周率')
    print(predict_sentence, predict_id)

    nmt_predict.predict_batch(PROJECT_PATH / 'data/test_data/positive.data')