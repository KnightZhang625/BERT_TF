# coding:utf-8

import sys
import logging
import tensorflow as tf
from pathlib import Path

PROJECT_PATH = Path(__file__).absolute().parent.parent
sys.path.insert(0, str(PROJECT_PATH))

class Setup(object):
    """Setup logging"""
    def __init__(self, log_name='tensorflow', path=str(PROJECT_PATH / 'log')):
        # excellent way to create directory
        Path('log').mkdir(exist_ok=True)
        # set the priority of the log level
        tf.compat.v1.logging.set_verbosity(logging.INFO)
        # create two handlers: one that will write the logs to sys.stdout(the tenminal windom),
        # and one to a file(as the FileHandler name implies).
        handlers = [logging.FileHandler(str(PROJECT_PATH / 'log/main.log')),
                    logging.StreamHandler(sys.stdout)]
        logging.getLogger('tensorflow').handlers = handlers