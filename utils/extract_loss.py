# coding:utf-8
# This file is used for extracting loss from the log file.

import re
import sys
import codecs

from log import log_info as _info
from log import log_error as _error

PATTERN = r'^(loss = )\d{1,5}\.\d{1,10}' 

def extract(log_path, save_path):
    with codecs.open(log_path, 'r', 'utf-8') as file, \
         codecs.open(save_path, 'w', 'utf-8') as file_2:
        for line in file:
            if re.search(PATTERN, line):
                match = re.search(PATTERN, line).group()
                loss = match.split(' ')[2]
                file_2.write('sup_avg:' + loss + '\n')
                file_2.flush()
    _info('The loss record have been save to {}.'.format(save_path))

if __name__ == '__main__':
    if len(sys.argv) < 3:
        _error('Please specify the log path and the save path.')
        raise ValueError
    else:
        log_path = sys.argv[1]
        save_path = sys.argv[2]
        extract(log_path, save_path)