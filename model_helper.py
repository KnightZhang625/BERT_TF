# coding:utf-8

import six
import tensorflow as tf

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
        raise ValueError(
            'For the tensor {} in scope {}, the tensor rank {%d} \
            (shape = {}) is not equal to the expected_rank {}'.format(
            name, scope_name, tensor_rank, str(tensor.shape), str(expected_rank)))

def create_initializer(initializer_range=0.02):
    return tf.truncated_normal_initializer(stddev=initializer_range)

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
                                        initializer=_mh.create_initializer(initializer_range))
    
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
    project_variable = tf.get_variable(name='projection_embeddings',
                                       shape=[embedding_size, hidden_size],
                                       initializer=create_initializer(initializer_range))
    # [batch_size, seq_length, embedding_size] -> [batch_size, seq_length, hidden_size]
    output = tf.matmul(embeddings, project_variable)

    return output, embedding_table, project_variable