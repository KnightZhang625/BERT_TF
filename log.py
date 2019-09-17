# coding: utf-8

from __future__ import print_function

def log_info(message, head='INFO'):
    print('\033[1;34m {} : {} \033[0m'.format(head, message))

def log_error(message, head='INFO'):
    print('\033[1;31m {} : {} \033[0m'.format(head, message))

if __name__ == '__main__':
    log_info('This is a test')
    log_info('user-defined test', head='USER')