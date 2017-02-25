import torch
from collections import defaultdict
import numpy as np
import os
import warnings

from .common import floatx
from .common import _EPSILON
from .common import image_data_format


def learning_phase():
    # False = test, True = train
    return _LEARNING_PHASE


def set_learning_phase(value):
    global _LEARNING_PHASE
    if value not in {0, 1}:
        raise ValueError('Expected learning phase to be '
                         '0 or 1.')
    _LEARNING_PHASE = value


def variable(value, dtype=None, name=None):
    """Instantiates a variable and returns it.

    # Arguments
        value: Numpy array, initial value of the tensor.
        dtype: Tensor type.
        name: Optional name string for the tensor.

    # Returns
        A variable instance (with Keras metadata included).
    """
    if dtype is None:
        dtype = floatx()
    if hasattr(value, 'tocoo'):
        raise NotImplementedError("sparse not implemented in this backend yet.")
        # _assert_sparse_module()
        # variable = th_sparse_module.as_sparse_variable(
        #     value, name=_prepare_name(name, 'variable'))
    else:
        # TODO: value from tensor
        if not isinstance(value, (torch.Tensor)):
            value = torch.from_numpy(value)

        variable = torch.autograd.Variable(value, requires_grad=True)  # TODO: name?
        # variable = theano.shared(value=value,
        #                          name=_prepare_name(name, 'variable'),
        #                          strict=False)
    variable._keras_shape = value.size()
    variable._uses_learning_phase = False
    return variable


def constant(value, dtype=None, shape=None, name=None):
    if dtype is None:
        dtype = floatx()
    if shape is None:
        shape = ()
    np_value = value * np.ones(shape)
    const = T.constant(np_value,
                       dtype=dtype,
                       name=_prepare_name(name, 'constant'))
    const._keras_shape = shape
    const._uses_learning_phase = False
    return const


def placeholder(shape=None, ndim=None, dtype=None, sparse=False, name=None):
    """Instantiate an input data placeholder variable.
    """
    if dtype is None:
        dtype = floatx()
    if shape is None and ndim is None:
        raise ValueError('Specify either a shape or ndim value.')
    if shape is not None:
        ndim = len(shape)
    else:
        shape = tuple([None for _ in range(ndim)])

    name = _prepare_name(name, 'placeholder')
    broadcast = (False,) * ndim
    if sparse:
        _assert_sparse_module()
        x = th_sparse_module.csr_matrix(name=name, dtype=dtype)
    else:
        x = T.TensorType(dtype, broadcast)(name)
    x._keras_shape = shape
    x._uses_learning_phase = False
    return x


def shape(x):
    """Returns the shape of a tensor.

    Warning: type returned will be different for
    Theano backend (Theano tensor type) and TF backend (TF TensorShape).
    """
    return x.size()


def int_shape(x):
    """Returns the shape of a Keras tensor or a Keras variable as a tuple of
    integers or None entries.

    # Arguments
        x: Tensor or variable.

    # Returns
        A tuple of integers (or None entries).
    """
    if hasattr(x, '_keras_shape'):
        return x._keras_shape
    else:
        raise TypeError('Not a Keras tensor:', x)


def ndim(x):
    return x.ndim


def dtype(x):
    return to_dtype(x.type())


def eval(x):
    """Returns the value of a tensor.
    """
    return x.numpy()


def get_value(x):
    # if not hasattr(x.data, ''):
    #     raise TypeError('get_value() can only be called on a variable. '
    #                     'If you have an expression instead, use eval().')
    return x.data.numpy()


def set_value(x, value):
    """Sets the value of a variable, from a Numpy array.

    # Arguments
        x: Tensor to set to a new value.
        value: Value to set the tensor to, as a Numpy array
            (of the same shape).
    """
    pass
    value = np.asarray(value)
    new = torch.from_numpy(value)
    x.data.set_(new.storage())
    #
    # tf_dtype = _convert_string_dtype(x.dtype.name.split('_')[0])
    # if hasattr(x, '_assign_placeholder'):
    #     assign_placeholder = x._assign_placeholder
    #     assign_op = x._assign_op
    # else:
    #     assign_placeholder = tf.placeholder(tf_dtype, shape=value.shape)
    #     assign_op = x.assign(assign_placeholder)
    #     x._assign_placeholder = assign_placeholder
    #     x._assign_op = assign_op
    # get_session().run(assign_op, feed_dict={assign_placeholder: value})


def to_torch_type(dtype):
    return {
        'ndarray': torch.Tensor,
        'float32': torch.FloatTensor,
        'float64': torch.DoubleTensor,
        'int8': torch.CharTensor,
        'uint8': torch.ByteTensor,
        'int16': torch.ShortTensor,
        'int32': torch.IntTensor,
        'int64': torch.LongTensor
    }[dtype]


def to_dtype(torch_type):
    return {
        'torch.Tensor': 'ndarray',
        'torch.FloatTensor': 'float32',
        'torch.DoubleTensor': 'float64',
        'torch.CharTensor': 'int8',
        'torch.ByteTensor': 'uint8',
        'torch.ShortTensor': 'int16',
        'torch.IntTensor': 'int32',
        'torch.LongTensor': 'int64'
    }[torch_type]


py_range = range


def arange(start, stop=None, step=1.0, dtype='int32'):
    """Creates a 1-D tensor containing a sequence of integers.

    The function arguments use the same convention as
    Theano's arange: if only one argument is provided,
    it is in fact the "stop" argument.

    The default type of the returned tensor is 'int32' to
    match TensorFlow's default.
    """
    if stop is None:
        stop = start
        start = 0.0
    if not np.arange(start, stop, step).size:
        return to_torch_type(dtype)()

    # Using epsilon to get an exclusive range operator from an inclusive one:
    stop = -np.signbit(step) * max(start, stop - _EPSILON) + np.signbit(step) * min(start, stop + _EPSILON)

    return torch.range(start, end=stop, step=float(step)).type(to_torch_type(dtype))
