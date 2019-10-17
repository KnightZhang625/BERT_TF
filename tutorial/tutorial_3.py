"""this file will show how to create pipeline according to our habits."""

import functools
import numpy as np
import tensorflow as tf
tf.enable_eager_execution()

import sys
import logging
from pathlib import Path

"""Setup logging"""
# excellent way to create directory
Path('results_3').mkdir(exist_ok=True)
# set the priority of the log level
tf.compat.v1.logging.set_verbosity(logging.INFO)
# create two handlers: one that will write the logs to sys.stdout(the tenminal windom),
# and one to a file(as the FileHandler name implies).
handlers = [
    logging.FileHandler('results_3/main.log'),
    logging.StreamHandler(sys.stdout)
]
logging.getLogger('tensorflow').handlers = handlers

class Model(object):
    def __init__(self, input_tensor):
        print(input_tensor)
        self.output = tf.layers.dense(input_tensor, 1)
    
    def get_output(self):
        return self.output

# define the model
def model_fn(features, labels, mode, params):
    """this is prototype syntax, all parameters are necessary."""
    if isinstance(features, dict):
        x = features['x_1']
        x_2 = features['x_2']
        labels = features['y']

    # input_tensor = tf.concat((x_1, x_2), axis=-1)
    # model = Model(input_tensor)
    # predictions = model.get_output()
 
    predictions = tf.layers.dense(x, 1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    else:
        loss = tf.nn.l2_loss(predictions - labels)
        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode, loss=loss)
        elif mode == tf.estimator.ModeKeys.TRAIN:
            train_op = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
        else:
            raise NotImplementedError()

def train_generator_fn(X_1, X_2, Y):
    for (x_1, x_2, y) in zip(X_1, X_2, Y):
        features = {'x_1': x_1, 'x_2': x_2, 'y': y}
        yield features

def train_input_fn(X_1, X_2, Y):
    output_types = {'x_1': tf.float32, 'x_2': tf.float32, 'y': tf.float32}
    output_shapes = {'x_1': (2), 'x_2': (1), 'y': (1)}
    dataset = tf.data.Dataset.from_generator(
       functools.partial(train_generator_fn, X_1, X_2, Y),
       output_types=output_types,
       output_shapes=output_shapes)
    dataset = dataset.batch(5).repeat(1000)
    return dataset

# def my_service():
#     for number in range(100, 110):
#         yield number

def serving_input_receiver_fn():
    x = tf.placeholder(dtype=tf.float32, shape=[None, 2], name='number')
    receiver_tensors = {'user_input': x}
    features = {'x_1': x, 'x_2': tf.zeros(0), 'y': tf.zeros(0)}
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

if __name__ == '__main__':
    # create some data, the function is f(x,y) = x + y
    x_1 = np.array([[i, i] for i in range(100)], dtype=np.float)               # (100, 3)
    x_2 = np.array([[i] for i in range(100, 200)], dtype=np.float)          # (100, 1)
    y = np.array([[2 * x[0]] for x in x_1], dtype=np.float)       # (100, )
    
    # # eager mode test
    # for features in train_input_fn(x_1, x_2, y):
    #     print(features['x_1'])
    #     print(features['x_2'])
    #     print(features['y'])
    #     input()

    # input_fn = functools.partial(train_input_fn, X_1=x_1, X_2=x_2, Y=y)
    # estimator = tf.estimator.Estimator(model_fn, 'model_3', params={})
    # estimator.train(input_fn)


    # estimator = tf.estimator.Estimator(model_fn, 'model_3', params={})
    # estimator.export_saved_model('saved_model_3', serving_input_receiver_fn)

    export_dir = 'saved_model_3'
    subdirs = [x for x in Path(export_dir).iterdir()
            if x.is_dir() and 'temp' not in str(x)]
    latest = str(sorted(subdirs)[-1])

    from tensorflow.contrib import predictor
    predict_fn = predictor.from_saved_model(latest)
    # for nb in my_service():
    pred = predict_fn({'user_input': np.array([[2, 2]], dtype=np.float)})['output']
    print(pred)