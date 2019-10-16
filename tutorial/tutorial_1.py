# encoding : utf-8

import sys
import codecs
import logging
import functools
from pathlib import Path

import tensorflow as tf
# tf.enable_eager_execution()
from tf_metrics import precision, recall, f1

"""Setup logging"""
# excellent way to create directory
Path('results').mkdir(exist_ok=True)
# set the priority of the log level
tf.compat.v1.logging.set_verbosity(logging.INFO)
# create two handlers: one that will write the logs to sys.stdout(the tenminal windom),
# and one to a file(as the FileHandler name implies).
handlers = [
    logging.FileHandler('results/main.log'),
    logging.StreamHandler(sys.stdout)
]
logging.getLogger('tensorflow').handlers = handlers

"""Feed data with tf.data
The standard technique was to use tf.placeholder that was updated through the run method of a tf.Session object.
Forget all this, because we won't use tf.Session anymore, cheer ! cheer !"""
# define a dummy data generator
"""
def generator_fn():
    for digit in range(2):
        line = 'I am digit {}'.format(digit)
        words = line.split()
        yield [w.encode() for w in words], len(words)

for words in generator_fn():
    print(words)
"""

# make the output of this generator available inside our graph,
# create a special Dataset node.
"""
shape = ([None], ())
types = (tf.string, tf.int32)
dataset = tf.data.Dataset.from_generator(generator_fn, output_shapes=shape, output_types=types)
"""
# Tensorflow provides other way of creating datasets, from text files(see tf.data.TextLineDataset),
# from np arrays(see tf.data.Dataset.from_tensor_slices), from TF records(see tf.data.TFRecordDataset), etc.
# For most NLP cases, I advise you to take advantages of the felxibility given by tf.data.from_generator
# unless you need the extra boost in performance provided by the other fancier options.

# Test tf.data pipeline
## 1. use the eager_execution mode, uncomment the tf.eager_execution() first.
"""
for tf_words, tf_size in dataset:
    print(tf_words, tf_size)
"""
## 2. use and old-school, not-so-user-friendly-but-still-usefull tf.Session, 
## apologize for the declaration that never use tf.Session above.
"""
iterator = dataset.make_one_shot_iterator()
node = iterator.get_next()
with tf.Session() as sess:
    while True:
        try:
            print(sess.run(node))
        except tf.errors.OutOfRangeError:
            break
"""

# Read from file and tokenize, we have 2 files words and tags,
# each line containing a white-spaced tokenized tagged sentence.
## the generator function, which reads the files and parse the lines
def parse_fn(line_words, line_tags):
    words = [w.encode() for w in line_words.strip().split()]
    tags = [t.encode() for t in line_tags.strip().split()]
    assert len(words) == len(tags), 'the length of words and tags must be identical.'
    return (words, len(words)), tags

def generator_fn(words, tags):
    with codecs.open(words, 'r', 'utf-8') as file_words,\
         codecs.open(tags, 'r', 'utf-8') as file_tags: 
        for line_words, line_tags in zip(file_words, file_tags):
            yield parse_fn(line_words, line_tags)

## the input_fn which constructs the dataset(needed by tf.estimator later)
def input_fn(words, tags, params=None, shuffle_and_repeat=False):
    params = params if params is not None else {}
    shapes = (([None], ()), [None])
    types = ((tf.string, tf.int32), tf.string)
    defaults = (('<pad>', 0), '0')

    dataset = tf.data.Dataset.from_generator(
        functools.partial(generator_fn, words, tags),
        output_shapes=shapes,
        output_types=types)
    
    if shuffle_and_repeat:
        dataset = dataset.shuffle(params['buffer']).repeat(params['epochs'])
    
    dataset = dataset.padded_batch(params.get('batch_size', 20), shapes, defaults).prefetch(1)

    return dataset

## Global Logic of the model_fn
def model_fn(features, labels, mode, params):
    # for serving, features are a bit different
    if isinstance(features, dict):
        features = features['word'], features['nwords']
    
    # Read vocabs and inputs
    dropout = params['dropout']
    words, nwords = features
    training = (mode == tf.estimator.ModeKeys.TRAIN)
    # convert words to idx
    vocab_words = tf.contrib.lookup.index_table_from_file(
        params['words'], num_oov_buckets=params['num_oov_buckets'])
    with codecs.open(params['tags'], 'r', 'utf-8') as file:
        indices = [idx for idx, tag in enumerate(file) if tag.strip() != '0']
        num_tags = len(indices) + 1
    
    # Word Embeddings
    word_ids = vocab_words.lookup(words)
    embedded = tf.get_variable('embedding', [params['vocab_size', 'embedding_size']], dtype=tf.float32)
    embeddings = tf.nn.embedding_lookup(embedded, word_ids)
    embeddings = tf.layers.dropout(embeddings, rate=dropout, training=training)

    # LSTM
    encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(params['num_units'])
    encoder_outputs, _ = tf.nn.dynamic_rnn(encoder_cell, 
                                                       embeddings, 
                                                       sequence_length=nwords, 
                                                       time_major=False)
    logits = tf.layers.dense(encoder_outputs, num_tags, name='projection')
    
    outputs = tf.nn.softmax(logits, axis=-1)
    pred_ids = tf.math.argmax(outputs, axis=-1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        # predictions
        reverse_vocab_tags = tf.contrib.lookup.index_to_string_table_from_file(params['tags'])
        pred_strings = reverse_vocab_tags.lookup(tf.to_int64(pred_ids))
        predictions = {
            'pred_ids': pred_strings,
            'tags': pred_strings
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    else:
        vocab_tags = tf.contrib.lookup.index_table_from_file(params['tags'])
        tags = vocab_tags.lookup(labels)
        loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tags, logits=logits)) / params['batch_size']
        tf.summary.scalar('loss', loss)

        weights = tf.sequence_mask(nwords)
        metrics = {
            'acc': tf.metrics.accuracy(tags, pred_ids, weights),
            'precision': precision(tags, pred_ids, num_tags, indices, weights),
            'recall': recall(tags, pred_ids, num_tags, indices, weights),
            'f1': f1(tags, pred_ids, num_tags, indices, weights)
        }

        for metric_name, op in metrics.items():
            tf.summary.scalar(metric_name, op[1])

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
        elif mode == tf.estimator.ModeKeys.TRAIN:
            train_op = tf.train.AdamOptimizer().minimize(loss, global_step=tf.train.get_or_create_global_step())
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
        else:
            raise NotImplementedError('Unknown mode {}'.format(mode))