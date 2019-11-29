# coding:utf-8
# Produced by Andysin Zhang
# 23_Oct_2019
# Inspired By the original Bert, Appreciate for the wonderful work
#
# Copyright 2019 TCL Inc. All Rights Reserverd.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

""""Run masked LM/next sentence masked_lm pre-training for ALBERT."""

import sys
import functools
import tensorflow as tf
# tf.enable_eager_execution()

from pathlib import Path
PROJECT_PATH = Path(__file__).absolute().parent
sys.path.insert(0, str(PROJECT_PATH))

from utils.setup import Setup
setup = Setup()

import optimization
from model import BertModel
import model_helper as _mh
from model_helper import *
from config import bert_config
from load_data_lm import train_input_fn, server_input_receiver_fn
from utils.log import log_info as _info
from utils.log import log_error as _error

def model_fn_builder(bert_config, init_checkpoint, learning_rate, num_train_steps):
    def model_fn(features, labels, mode, params):
        _info('*** Features ***')
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features['input_ids']       # [batch_size, seq_length]

        # build model
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        model = BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=input_ids)
   
        # [b, s, h]
        sequence_output = model.get_sequence_output()
        sequence_output = tf.reshape(sequence_output, 
                                [-1, bert_config.max_length * bert_config.hidden_size])
        
        with tf.variable_scope('prediction'):
            logits  = tf.layers.dense(sequence_output, 
                                  bert_config.classes,
                                  name='prediction',
                                  kernel_initializer=_mh.create_initializer(0.2))
            prob = tf.nn.softmax(logits, axis=-1)       # [b, 2]
            predict_ids = tf.argmax(prob, axis=-1)    # [b, ]

            if mode == tf.estimator.ModeKeys.PREDICT:
                predictions = {'class': predict_ids}
                # the default key in 'output', however, when customized, the keys are identical with the keys in dict.
                output_spec = tf.estimator.EstimatorSpec(mode, predictions=predictions)
            else:
                if mode == tf.estimator.ModeKeys.TRAIN:
                    tvars = tf.trainable_variables()
                    initialized_variable_names = {}
                    if init_checkpoint:
                        (assignment_map, initialized_variable_names) = get_assignment_map_from_checkpoint(tvars, init_checkpoint)
                        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

                    _info('*** Trainable Variables ***')
                    for var in tvars:
                        init_string = ''
                        if var.name in initialized_variable_names:
                            init_string = ', *INIT_FROM_CKPT*'
                        _info('name = {}, shape={}{}'.format(var.name, var.shape, init_string))


                    batch_size = tf.cast(bert_config.batch_size, tf.float32) 

                    labels = tf.reshape(labels, [-1])
    
                    # logits = tf.expand_dims(logits, axis=1)
                    seq_loss = tf.reduce_sum(
                            tf.nn.sparse_softmax_cross_entropy_with_logits(
                                labels=labels, logits=logits)) / batch_size
                    loss = seq_loss
                    """
                    Tutorial on `polynomial_decay`:
                        The formula is as below:
                            
                            global_step = min(global_step, decay_steps)
                            decayed_learning_rate = (learning_rate - end_learning_rate) * (1 - global_step / decay_steps) ^ (power) + end_learning_rate
                        
                        global_step: each batch step.
                        decay_steps: the whole step, the lr will touch the end_learning_rate after the decay_steps.
                        TRAIN_STEPS: the number for repeating the whole dataset, so the decay_steps = len(dataset) / batch_size * TRAIN_STEPS.
                    """
                    train_op, lr = optimization.create_optimizer(loss, bert_config.learning_rate, bert_config.num_train_steps * 5, bert_config.lr_limit)
                    """
                    learning_rate = tf.train.polynomial_decay(config.learning_rate,
                                                            tf.train.get_or_create_global_step(),
                                                            _cg.TRIAN_STEPS,
                                                            end_learning_rate=0.0,
                                                            power=1.0,
                                                            cycle=False)

                    lr = tf.maximum(tf.constant(config.lr_limit), learning_rate)
                    optimizer = tf.train.AdamOptimizer(lr, name='optimizer')
                    tvars = tf.trainable_variables()
                    gradients = tf.gradients(loss, tvars, colocate_gradients_with_ops=config.colocate_gradients_with_ops)
                    clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
                    train_op = optimizer.apply_gradients(zip(clipped_gradients, tvars), global_step=tf.train.get_global_step())
                    """

                    # this is excellent, because it could display the result each step, i.e., each step equals to batch_size.
                    # the output_spec, display the result every save checkpoints step.
                    logging_hook = tf.train.LoggingTensorHook({'loss' : loss, 'lr': lr}, every_n_iter=10)

                    output_spec = tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, training_hooks=[logging_hook])

                elif mode == tf.estimator.ModeKeys.EVAL:
                    # TODO
                    raise NotImplementedError
            
            return output_spec
        
    return model_fn

def main():
    Path(bert_config.model_dir).mkdir(exist_ok=True)

    model_fn = model_fn_builder(bert_config, bert_config.init_checkpoint, bert_config.learning_rate, bert_config.num_train_steps)

    input_fn = functools.partial(train_input_fn, 
                                path=bert_config.data_path,
                                batch_size=bert_config.batch_size,
                                repeat_num=bert_config.num_train_steps,
                                max_length = bert_config.max_length)

    run_config = tf.contrib.tpu.RunConfig(
        keep_checkpoint_max=1,
        save_checkpoints_steps=10,
        model_dir=bert_config.model_dir)
    
    estimaotr = tf.estimator.Estimator(model_fn, config=run_config)
    estimaotr.train(input_fn)     # train_input_fn should be callable

def package_model(ckpt_path, pb_path):
    model_fn = model_fn_builder(bert_config, None, bert_config.learning_rate, bert_config.num_train_steps)
    estimator = tf.estimator.Estimator(model_fn, ckpt_path)
    estimator.export_saved_model(pb_path, server_input_receiver_fn)

if __name__ == '__main__':
    if sys.argv[1] == 'train':
        main()
    elif sys.argv[1] == 'package':
        package_model(str(PROJECT_PATH / 'models_lm'), str(PROJECT_PATH / 'models_deploy_lm'))
    else:
        _error('Unknown parameter: {}.'.format(sys.argv[1]))
        _info('Choose from [train | package].')