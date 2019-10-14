# coding:utf-8
# used for executing the UniLM model

import os
import argparse
import collections
import tensorflow as tf
import model_helper as _mh

from pathlib import Path
from model import BertModel
from hparams_config import config as config_
from log import log_info as _info
from log import log_error as _error

PROJECT_PATH = str(Path(__file__).absolute().parent)

def train(config):
