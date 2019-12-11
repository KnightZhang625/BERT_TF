# coding:utf-8

import re
import sys
import six
import collections
import numpy as np
import tensorflow as tf

from pathlib import Path
PROJECT_PATH = Path(__file__).absolute().parent
sys.path.insert(0, str(PROJECT_PATH))

from utils.log import log_info as _info
from utils.log import log_error as _error

"""TENSOR CALCULATE"""
def get_shape_list(tensor, expected_rank=None, name=None):
    """Returns a list of shape of tensor, preferring static dimensions, 
        Sometimes, the dimension is None.
        
    Args:
        tensor: A tf.Tensor which needs to find the shape.
        expected_rand: (optional) int. The expected rank of 'tensor'. If this is
            specified and the 'tensor' has a different rank, an error will be thrown.
        name: (optional) name of the 'tensor' when throwing the error.
    
    Returns:
        A list of dimensions of the shape of the tensor.
        All static dimensions will be returned as python integers,
        and dynamic dimensions will be returned as tf.Tensir scalars.
    """
    if name is None:
         name = tensor.name
        
    if expected_rank is not None:
        assert_rank(tensor, expected_rank, name)
    
    shape = tensor.shape.as_list()

    # save the dimension which is None
    non_static_indices = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indices.append(index)
    
    # non_static_indices is empty
    if not non_static_indices:
        return shape
    
    # non_static_indices saves the index of dynamic shape,
    # replace those dynamic shapes in the shape list.
    dynamic_shape = tf.shape(tensor)
    for index in non_static_indices:
        shape[index] = dynamic_shape[index]
    return shape

def assert_rank(tensor, expected_rank, name=None):
    """Check whether the rank of the 'tensor' matches the expected_rank.
        Remember rank is the number of the total dimensions.
    
    Args:
        tensor: A tf.Tensor to check.
        expected_rank: Python integer or list of intefers.
        name: (optional) name for the error.
    """
    if name is None:
        name = tensor.name
    
    expected_rank_dict = {}
    # save the given rank into the dictionary
    if isinstance(expected_rank, six.integer_types):
        expected_rank_dict[expected_rank] = True
    else:
        for rank in expected_rank:
            expected_rank_dict[rank] = True
    
    tensor_rank = tensor.shape.ndims
    if tensor_rank not in expected_rank_dict:
        scope_name = tf.get_variable_scope().name
        _error( 'For the tensor {} in scope {}, the tensor rank {%d} \
            (shape = {}) is not equal to the expected_rank {}'.format(
            name, scope_name, tensor_rank, str(tensor.shape), str(expected_rank)))
        raise ValueError

def create_initializer(initializer_range=0.02):
    return tf.truncated_normal_initializer(stddev=initializer_range)

def layer_norm(input_tensor, name=None):
    return tf.contrib.layers.layer_norm(inputs=input_tensor, scope=None)

def batch_norm(input_tensor, is_training):
  return tf.contrib.layers.batch_norm(input_tensor, scale=True, is_training=is_training)

def dropout(input_tensor, dropout_prob):
    if dropout_prob is None or dropout_prob == 0.0:
        return input_tensor
    
    output = tf.nn.dropout(input_tensor, keep_prob=1.0 - dropout_prob)
    return output

def layer_norm_and_dropout(input_tensor, dropout_prob, name=None):
    output_tensor = layer_norm(input_tensor, name)
    output_tensor = dropout(input_tensor, dropout_prob)
    return output_tensor
  
def gelu(x):
    """Gaussian Error Linear Unit."""
    cdf = 0.5 * (1.0 + tf.tanh(
        (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf

"""FOR EMBEDDING"""
def embedding_lookup_factorized(input_ids,
                                vocab_size,
                                hidden_size,
                                embedding_size,
                                use_one_hot_embedding,
                                initializer_range=0.02,
                                word_embedding_name='albert_word_embeddings'):
    """create albert embeddings, which reduce the number of the prameters, V -> E, E -> H, where E < H.

    Args:
        input_ids: int32 Tensor of shape [batch_size, seq_length].
        vocab_size: int. Size of the vocabulary.
        embedding_size: int. Dimension for the word embeddings.
        initializer_type: float.
        word_embedding_name: string.
        use_one_hot_embeddings: bool.
    """
    # create word embeddings
    embedding_table = tf.get_variable(name=word_embedding_name, 
                                        shape=[vocab_size, embedding_size], 
                                        initializer=create_initializer(initializer_range))
    
    # calculate word embeddings
    # the embeddings shape is [batch_size, seq_length, embedding_size]
    if use_one_hot_embedding:
        # [batch_size, seq_length] -> [batch_size, seq_length, vocab_size]
        one_hot_input_ids = tf.one_hot(input_ids, depth=vocab_size)
        # [batch_size, seq_length, embedding_size]
        embeddings = tf.matmul(one_hot_input_ids, embedding_table)
    else:
        embeddings = tf.nn.embedding_lookup(embedding_table, input_ids)
    
    # project embedding dimension to the hidden size
    project_variable = tf.get_variable(name=word_embedding_name+'_2',
                                       shape=[embedding_size, hidden_size],
                                       initializer=create_initializer(initializer_range))
    # [batch_size, seq_length, embedding_size] -> [batch_size, seq_length, hidden_size]
    output = tf.matmul(embeddings, project_variable)

    return output, embedding_table, project_variable

def embedding_postprocessor(input_tensor,
                            use_token_type=False,
                            token_type_ids=None,
                            token_type_vocab_size=3,
                            token_type_embedding_name='token_type_embeddings',
                            use_positional_embeddings=True,
                            positional_embedding_type='normal',
                            pre_positional_embeddings=None,
                            positional_embedding_name='position_embeddings',
                            initializer_range=0.01,
                            max_positional_embeddings=512,
                            dropout_prob=0.01):
    """Performs some preprocessing on the word embeddings.
    
    Args:
        input_tensor: float Tensor of shape [batch_size, seq_length, embedding_size].
        use_token_type: bool. Whether to add segment embeddings, very confused about the original comments
            uses 'token' as name, as I realized, token_type_ids would be [[0, 0, 1], [0, 1, 0]], 0 refers to the segment 1,
            and 1 refers to segment 2, the last 0 in the second array refers to the padding.
        token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
        token_type_vocab_size: the number of token types.
        use_positional_embeddings: bool. Whether to add positional embeddings.
        positional_embedding_type: ['normal', 'trigonometrical'].
        pre_positional_embeddings: postional embeddings for the pre_positional_embeddings.
        postional_embedding_name: string. The name of the embedding table variable.
        initializer_range: float. Range of the weight initializer.
        max_positional_embeddings: int. Maximum sequence length for each sentence, which should be equal to or longer than the sequence.
        dropout_prob: float. Dropout probability applied to the final output tensor.
    
    Returns:
        float Tensor with the identical shape as 'input_tensor'.
    """
    input_shape = get_shape_list(input_tensor, expected_rank=[2,3])
    batch_size, seq_length, width = input_shape[0], input_shape[1], input_shape[2]

    # create this variable in case of not use any pre-embeddings on the input_tensor
    output = input_tensor

    if use_token_type:
        if token_type_ids is None:
            _error('`token_type_ids` must be specified if `use_token_type` is True.')
            raise ValueError
        token_type_table = tf.get_variable(
            name=token_type_embedding_name,
            shape=[token_type_vocab_size, width],
            initializer=create_initializer(initializer_range))
        
        token_type_embeddings = tf.nn.embedding_lookup(token_type_table, token_type_ids)
        output += token_type_embeddings

    if use_positional_embeddings:
        assert_op = tf.assert_less_equal(seq_length, max_positional_embeddings)
        with tf.control_dependencies([assert_op]):
            full_positional_embeddings = tf.get_variable(
                name=positional_embedding_name,
                shape=[max_positional_embeddings, width],
                initializer=create_initializer(initializer_range))
            
            # the full_positional_embeddings is created under the maximum sequence length,
            # however, the actual length maybe less than the maximum length, so slicing is necessary.
            positional_embeddings = tf.slice(full_positional_embeddings, [0, 0], [seq_length, -1])
            output += positional_embeddings
    
    output = layer_norm_and_dropout(output, dropout_prob)
    return output

"""FOR MASK"""
def create_attention_mask_from_input_mask(input_ids, input_mask):
    """create attention mask.
    
    Args:
        input_ids: int32 Tensor of shape [batch_size, seq_length].
        input_mask: int32 Tensor of shape [batch_size, seq_length].
    
    Returns:
        float Tensor of shape [batch_size, seq_length, seq_length].
    """

    input_ids_shape = get_shape_list(input_ids, expected_rank=[2, 3])
    batch_size, input_length = input_ids_shape[0], input_ids_shape[1]

    input_mask_shape = get_shape_list(input_mask, expected_rank=[2, 3])
    mask_length = input_mask_shape[1]

    # initial_mask: [b, s, 1]       input_mask: [b, 1, s]
    #       1    1    1                 
    #       1    1    1                 b1: 1 1 0 1 1
    #   b1: 1 b2:1 b3:1                 b2: 1 1 0 1 0
    #       1    1    1                 b3: 1 1 1 0 1
    #       1    1    1   
    #                       1 1 0 1 1     1 1 0 1 0     1 1 1 0 1
    #                       1 1 0 1 1     1 1 0 1 0     1 1 1 0 1
    # mask: [b, s, s]-> b1: 1 1 0 1 1  b2:1 1 0 1 0  b3:1 1 1 0 1 
    #                       1 1 0 1 1     1 1 0 1 0     1 1 1 0 1
    #                       1 1 0 1 1     1 1 0 1 0     1 1 1 0 1
    #

    input_mask = tf.cast(
        tf.reshape(input_mask, [batch_size, 1, mask_length]), dtype=tf.float32)
    initial_input = tf.ones([batch_size, input_length, 1], dtype=tf.float32)
    
    mask = initial_input * input_mask
    return mask

"""FOR RESTORE"""
def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
    """Compute the union of the current variables and checkpoint variables."""
    assignment_map = {}
    initialized_variable_names = {}

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var

    init_vars = tf.train.list_variables(init_checkpoint)

    assignment_map = collections.OrderedDict()
    for x in init_vars:
        (name, var) = (x[0], x[1])
        if name not in name_to_variable:
            continue
        assignment_map[name] = name
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ":0"] = 1

    return (assignment_map, initialized_variable_names)

"""MODEL BLOCK"""
def vae(input_tensor, scope_name='vae'):
  """For VAE compute.
  
  Args:
    input_tensor: tf.float32 tensor, expected 2 ranks.
  """
  axis_1, axis_2 = get_shape_list(vae, expected_rank=2)
  
  with tf.variable_scope(scope_name):
    mu = tf.layers.dense(input_tensor, axis_1, name='{}_mu'.format(scope_name))
    logVar = tf.layers.dense(input_tensor, axis_2, name='{}_log_var'.format(scope_name))
    sigma = tf.exp(logVar / 2.0)
    epsilon = tf.random_normal([axis_1, axis_2])
    z = mu + sigma * epsilon
  
  return z

def kl_loss(logVar, mu, kl_tolerance=0.5):
  """Compute KL divergence."""
  _, axis_2 = get_shape_list(logVar)

  kl_loss = -0.5 * tf.reduce_sum(
    (1 + logVar - tf.square(mu) - tf.exp(logVar)),
    reduction_indices=1)
  kl_loss = tf.maximum(kl_loss, kl_tolerance * axis_2)
  kl_loss = tf.reduce_mean(kl_loss)
  
  return kl_loss