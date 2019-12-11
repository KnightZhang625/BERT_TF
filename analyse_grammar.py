# coding:utf-8

import sys
import codecs
import jieba.posseg as pseg

from pathlib import Path
PROJECT_PATH = Path(__file__).absolute().parent
sys.path.insert(0, str(PROJECT_PATH))

from utils.log import log_info as _info
from utils.log import log_error as _error

def analyse(path):
  with codecs.open(path, 'r', 'utf-8') as file:
    for line in file:
      result = pseg.cut(line.strip())
      print(list(result))
      input()
  
if __name__ == '__main__':
  analyse(PROJECT_PATH / 'data/test_data/positive.data')
