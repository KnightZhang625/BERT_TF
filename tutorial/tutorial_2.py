"""train an estimator to compute f([x,x])=2x
"""

import functools
import tensorflow as tf


import sys
import codecs
import logging
from pathlib import Path

"""Setup logging"""
# excellent way to create directory
Path('results_2').mkdir(exist_ok=True)
# set the priority of the log level
tf.compat.v1.logging.set_verbosity(logging.INFO)
# create two handlers: one that will write the logs to sys.stdout(the tenminal windom),
# and one to a file(as the FileHandler name implies).
handlers = [
    logging.FileHandler('results_2/main.log'),
    logging.StreamHandler(sys.stdout)
]
logging.getLogger('tensorflow').handlers = handlers

# <1> Train the Model
# def model_fn(features, labels, mode, params):
#     if isinstance(features, dict):  # For serving
#         features = features['feature']
    
#     predictions = tf.layers.dense(features, 1)

#     if mode == tf.estimator.ModeKeys.PREDICT:
#         return tf.estimator.EstimatorSpec(mode, predictions=predictions)
#     else:
#         loss = tf.nn.l2_loss(predictions - labels)
#         if mode == tf.estimator.ModeKeys.EVAL:
#             return tf.estimator.EstimatorSpec(mode, loss=loss)
#         elif mode == tf.estimator.ModeKeys.TRAIN:
#             train_op = tf.train.AdamOptimizer(learning_rate=0.5).minimize(loss, global_step=tf.train.get_global_step())
#             return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
#         else:
#             raise NotImplementedError()

# def train_generator_fn():
#     for number in range(100):
#         yield [number, number], [2 * number]

# def train_input_fn():
#     shapes, types = (2, 1), (tf.float32, tf.float32)
#     dataset = tf.data.Dataset.from_generator(
#         train_generator_fn, output_types=types, output_shapes=shapes)
#     dataset = dataset.batch(5).repeat(200)
#     return dataset

# estimator = tf.estimator.Estimator(model_fn, 'model', params={})
# estimator.train(train_input_fn)

# <2> Reload and Predict
def my_service():
    for number in range(100, 110):
        yield number

# ## we don't know the input as a web server, so use lambda to create fake generator
# def example_input_fn(number):
#     dataset = tf.data.Dataset.from_generator(
#         lambda : ([number, number] for _ in range(1)),
#         output_types=tf.float32, output_shapes=(2,))
#     iterator = dataset.batch(1).make_one_shot_iterator()
#     next_element = iterator.get_next()
#     return next_element, None

# ## predict as server
# for nb in my_service():
#     example_inpf = functools.partial(example_input_fn, nb)
#     for pred in estimator.predict(example_inpf):
#         print(pred)

"""However, every time we call predict, out estimator instance reloads the weights from disk(see main.log) from results_2 directory.
The following will introduce a better option, even though the guides and documentation are pretty scarce and vague about this,
exciting will come while reading the following."""

# def serving_input_receiver_fn():
#     number = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='number')
#     receiver_tensors = {'number': number}
#     features = tf.tile(number, multiples=[1, 2])
#     return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

# estimator = tf.estimator.Estimator(model_fn, 'model', params={})
# estimator.export_saved_model('saved_model', serving_input_receiver_fn)

export_dir = 'saved_model'
subdirs = [x for x in Path(export_dir).iterdir()
           if x.is_dir() and 'temp' not in str(x)]
latest = str(sorted(subdirs)[-1])

from tensorflow.contrib import predictor
predict_fn = predictor.from_saved_model(latest)
for nb in my_service():
    pred = predict_fn({'number': [[nb]]})['output']