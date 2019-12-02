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
from model_new import TextCNN
import model_helper as _mh
from model_helper import *
from cnn_config import cnn_config
from load_data_lm import train_input_fn, server_input_receiver_fn
from utils.log import log_info as _info
from utils.log import log_error as _error

def model_fn_builder(cnn_config):
    def model_fn(features, labels, mode, params):
        _info('*** Features ***')
        for name in sorted(features.keys()):
            tf.logging.info('   name = %s, shape = %s'%(name, features[name].shape))
        
        input_ids = features['input_ids']

        # build model
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        model = TextCNN(input_ids, is_training, cnn_config)

        output = model.get_output()
        
        with tf.variable_scope('prediction'):
            output_prob = tf.nn.softmax(output, axis=-1)
            predictions = tf.argmax(output_prob, axis=-1)
        
        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {'class': predictions}
            output_spec = tf.estimator.EstimatorSpec(mode, predictions=predictions)
        else:
            if mode == tf.estimator.ModeKeys.TRAIN:
                tvars = tf.trainable_variables()
                batch_size = tf.cast(cnn_config.batch_size, tf.float32)
  
                labels = tf.reshape(labels, [-1])

                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=labels)
                l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tvars]) * cnn_config.l2_reg_lambda
                loss = tf.reduce_sum(cross_entropy + l2_losses) / batch_size

                train_op, lr = optimization.create_optimizer(loss, cnn_config.learning_rate, cnn_config.num_train_steps * 5, cnn_config.lr_limit)
                
                logging_hook = tf.train.LoggingTensorHook({'loss': loss, 'lr': lr}, every_n_iter=10)
                
                output_spec = tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, training_hooks=[logging_hook])
            elif mode == tf.estimator.ModeKeys.EVAL:
                raise NotImplementedError
    
        return output_spec
    
    return model_fn

def main():
    Path(cnn_config.model_dir).mkdir(exist_ok=True)

    model_fn = model_fn_builder(cnn_config)

    input_fn = functools.partial(train_input_fn, 
                                path=cnn_config.data_path,
                                batch_size=cnn_config.batch_size,
                                repeat_num=cnn_config.num_train_steps,
                                max_length = cnn_config.max_length)

    run_config = tf.contrib.tpu.RunConfig(
        keep_checkpoint_max=1,
        save_checkpoints_steps=10,
        model_dir=cnn_config.model_dir)
    
    estimaotr = tf.estimator.Estimator(model_fn, config=run_config)
    estimaotr.train(input_fn)     # train_input_fn should be callable

def package_model(ckpt_path, pb_path):
    model_fn = model_fn_builder(cnn_config)
    estimator = tf.estimator.Estimator(model_fn, ckpt_path)
    estimator.export_saved_model(pb_path, server_input_receiver_fn)

if __name__ == '__main__':
    if sys.argv[1] == 'train':
        main()
    elif sys.argv[1] == 'package':
        package_model(str(PROJECT_PATH / 'models_cnn'), str(PROJECT_PATH / 'models_deploy_cnn'))
    else:
        _error('Unknown parameter: {}.'.format(sys.argv[1]))
        _info('Choose from [train | package].')