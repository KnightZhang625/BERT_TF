# coding:utf-8

import os
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
    # build the training graph
    train_graph = tf.Graph()
    with train_graph.as_default():
        train_model = BertModel(config=config, is_training=True)
        _info('finish building graph', head='INFO')
    
    # create session
    # the relationship between graph and session,
    # like python language and python interpreter.
    sess_conf = tf.ConfigProto(intra_op_parallelism_threads=8, inter_op_parallelism_threads=8)
    sess_conf.gpu_options.allow_growth = True
    train_sess = tf.Session(config=sess_conf, graph=train_graph)

    # restore model from the latest checkpoint
    with train_graph.as_default():
        loaded_model, global_step = _mh.create_or_load(
            train_model, os.path.join(PROJECT_PATH, config.ckpt_path), train_sess)
    
    # initialize Tensorboard
    summary_writer = tf.summary.FileWriter(
        os.path.join(config.ckpt_path, config.summary_name), train_graph)
    
    # train
    train_data = collections.namedtuple('data', 'input_ids input_length input_mask output_ids positional_embeddings')
    iter_steps = config.steps

    # the following code is just for test
    """
    import numpy as np
    input_ids = np.array([[10, 128, 10, 0, 120], [20, 3, 0, 0, 30]], dtype=np.int32)
    input_length = 5
    input_mask = np.array([[1, 1, 1, 0, 1], [1, 1, 0, 0, 1]], dtype=np.int32)
    output_ids = np.array([[10, 128, 10, 1, 120], [20, 3, 2, 5, 30]], dtype=np.int32)
    train_data.input_ids = input_ids
    train_data.input_length = input_length
    train_data.input_mask = input_mask
    train_data.output_ids = output_ids
    train_data.positional_embeddings = _mh.create_pos_embeddings(config.embedding_size, input_length)
    """

    while int(global_step) < int(iter_steps):
        for _ in range(100):
            result = train_model.train(train_sess, train_data)
            global_step, learning_rate, loss_bs, _, _, summary = \
                result[0], result[1], result[2], result[3], result[4], result[5]
            
            summary_writer.add_summary(summary, global_step)
            _info('loss : {}\t lr : {}'.format(loss_bs, learning_rate), head='Step: {}'.format(global_step))

            if global_step % 100 == 0:
                train_model.saver.save(
                    train_sess,
                    os.path.join(PROJECT_PATH, config.ckpt_path),
                    global_step=global_step)

def infer(config):
    # build graph
    infer_graph = tf.Graph()
    with infer_graph.as_default():
        infer_model = BertModel(config=config, is_training=False)
    
    # create session
    sess_conf = tf.ConfigProto(intra_op_parallelism_threads=8, inter_op_parallelism_threads=8)
    sess_conf.gpu_options.allow_growth = True
    infer_sess = tf.Session(config=sess_conf, graph=infer_graph)

    # restore model from the latest checkpoint
    with infer_graph.as_default():
        loaded_model, global_step = _mh.create_or_load(
            infer_model, os.path.join(PROJECT_PATH, config.ckpt_path), infer_sess)

    # the following is just for test
    """
    import numpy as np
    input_ids = np.array([[10, 128, 10, 0, 120], [20, 3, 0, 0, 30]], dtype=np.int32)
    input_length = 5
    input_mask = np.array([[1, 1, 1, 0, 1], [1, 1, 0, 0, 1]], dtype=np.int32)
   
    infer_data = collections.namedtuple('data', 'input_ids input_length input_mask positional_embeddings')
    infer_data.input_ids = input_ids
    infer_data.input_length = input_length
    infer_data.input_mask = input_mask
    infer_data.positional_embeddings = _mh.create_pos_embeddings(config.embedding_size, input_length)
    """

    # infer
    prediciton = infer_model.infer(infer_sess, infer_data)
    return prediciton

if __name__ == '__main__':
    train(config_)
    print(infer(config_))