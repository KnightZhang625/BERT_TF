"""this file will show how to create pipeline according to our habits."""

import functools
import numpy as np
import tensorflow as tf
# tf.enable_eager_execution()

import sys
import logging
from pathlib import Path

"""Setup logging"""
# excellent way to create directory
Path('result').mkdir(exist_ok=True)
# set the priority of the log level
tf.compat.v1.logging.set_verbosity(logging.INFO)
# create two handlers: one that will write the logs to sys.stdout(the tenminal windom),
# and one to a file(as the FileHandler name implies).
handlers = [
    logging.FileHandler('result/main.log'),
    logging.StreamHandler(sys.stdout)
]
logging.getLogger('tensorflow').handlers = handlers

class Model(object):
    """Toy Model"""
    def __init__(self, x, b):
        self.output = tf.add(tf.layers.dense(x, 1), b)

    def get_output(self):
        return self.output

# define the model
def model_fn(features, labels, mode, params):
    """this is prototype syntax, all parameters are necessary."""
    # Send the data as a dictionary is a better choice
    if isinstance(features, dict):  
        x = features['x']
        b = features['b']
        labels = features['y']

    # this tests the params parameters
    batch_size = tf.cast(params['batch_size'], tf.float32)
    model = Model(x, b)
    outputs = model.get_output()

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=outputs)
    else:
        loss = tf.nn.l2_loss(outputs - labels) / batch_size
        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode, loss=loss)
        elif mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
            parameters = tf.trainable_variables()
            gradients = tf.gradients(loss, parameters, colocate_gradients_with_ops=True)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
            train_op = optimizer.apply_gradients(zip(clipped_gradients, parameters), global_step=tf.train.get_global_step())
            # train_op = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
        else:
            raise NotImplementedError()

def train_generator_fn(xs, bs, ys):
    """This is the entrance to the input_fn."""
    for (x, b, y) in zip(xs, bs, ys):
        # the key in the following dictionary
        # should be identical to the key of the output_types and output_shapes
        # in the train_input_fn.
        features = {'x': x, 'b': b, 'y': y}
        yield features

def train_input_fn(xs, bs, ys):
    output_types = {'x': tf.float32, 'b': tf.float32, 'y': tf.float32}
    output_shapes = {'x': (2), 'b': (1), 'y': (1)}
    dataset = tf.data.Dataset.from_generator(
       functools.partial(train_generator_fn, xs, bs, ys),
       output_types=output_types,
       output_shapes=output_shapes)
    dataset = dataset.batch(2).repeat(1000)
    return dataset

# def my_service():
#     for number in range(100, 110):
#         yield number

def serving_input_receiver_fn():
    x = tf.placeholder(dtype=tf.float32, shape=[None, 2], name='x')
    b = tf.placeholder(dtype=tf.float32, shape=[1], name='b')
    receiver_tensors = {'x': x, 'b': b}
    features = {'x': x, 'b': b, 'y': tf.zeros(0)}
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

if __name__ == '__main__':
    # create some data, the function is f(x,y) = x + y
    xs = [[2, 3], [3, 4], [4, 5], [5, 6], [6, 7]]
    bs = [[0.1] for _ in range(5)]
    ys = [[x[0] + 2 * x[1] + b[0]]for x, b in zip(xs, bs)]
    
    xs = np.array(xs, dtype=np.float)   # (5, 2)
    bs = np.array(bs, dtype=np.float)   # (5, 1)
    ys = np.array(ys, dtype=np.float)     # (5, 1)
    
    # # eager mode test
    # for features in train_input_fn(x_1, x_2, y):
    #     print(features['x_1'])
    #     print(features['x_2'])
    #     print(features['y'])
    #     input()

    # this is for training
    # input_fn = functools.partial(train_input_fn, xs=xs, bs=bs, ys=ys)
    # estimator = tf.estimator.Estimator(model_fn, 'model', params={'batch_size':2})
    # estimator.train(input_fn, steps=1000)


    # # load the model and export it to the pb format
    # estimator = tf.estimator.Estimator(model_fn, 'model', params={'batch_size':2})
    # estimator.export_saved_model('saved_model', serving_input_receiver_fn)

    # for user input
    export_dir = 'saved_model'
    subdirs = [x for x in Path(export_dir).iterdir()
            if x.is_dir() and 'temp' not in str(x)]
    latest = str(sorted(subdirs)[-1])

    from tensorflow.contrib import predictor
    predict_fn = predictor.from_saved_model(latest)
    i_x = np.array([[2, 3]], dtype=np.float)
    i_b = np.array([0.1], dtype=np.float)
    pred = predict_fn({'x': i_x, 'b': i_b})['output']
    print(pred)