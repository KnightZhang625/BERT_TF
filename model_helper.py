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
