# coding: utf-8

from __future__ import print_function

import sys

def log_info(message, head='INFO'):
    print('\033[1;34m {} : {} \033[0m'.format(head, message))

def log_error(message, head='INFO'):
    print('\033[1;31m {} : {} \033[0m'.format(head, message))

def print_process(percent):
    i = int(percent)
    sys.stdout.write("\r{0}{1}".format("|"*i , '%.2f%%' % (percent)))
    sys.stdout.flush()

if __name__ == '__main__':
    log_info('This is a test')
    log_info('user-defined test', head='USER')