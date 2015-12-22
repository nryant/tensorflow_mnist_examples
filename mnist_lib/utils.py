"""Miscellaneous utility functions.
"""
import numpy as np;
import tensorflow as tf;

__all__ = ['get_randn_variable', 'def_get_zeros_variable',
           'init_uninit', 'ndim', 'shape', 'size'];


def get_randn_variable(name, shape, dtype=tf.float32, mu=0, sigma=1):
    """TODO
    """
    return tf.get_variable(name, shape, dtype,
                           tf.truncated_normal_initializer(mu, sigma));


def get_zeros_variable(name, shape, dtype=tf.float32):
    """TODO
    """
    return tf.get_variable(name, shape, dtype,
                           tf.constant_initializer(0));


def init_uninit(var_list, session=None):
    """Initialize any variables in ``var_list`` that have not been
    initialized.

    Parameters
    ----------
    var_list : list of tensorflow.Variable
        List of tensorflow variables, which may be a mixture of
        uninitialized variables and initialized variables. Any
        uninitialized variables will be initialized.

    session : TODO
    """
    if session is None:
        session = tf.get_default_session();

    vars_to_init = [];
    for var in var_list:
        # NOTE: Currently, this prints a traceback whether we want it
        #       or not.
        try:
            session.run(tf.assert_variables_initialized([var]));
        except tf.errors.FailedPreconditionError:
            vars_to_init.append(var);
    init = tf.initialize_variables(vars_to_init);
    session.run(init);


def ndim(x):
    """Return number of dimensions of tensor.
    """
    return len(shape(x));


def shape(x):
    """Return shape of tensor.
    """
    return x.get_shape();


def size(x):
    """Return size of tensor.
    """
    shape = x.get_shape();
    if None in shape:
        return None;
    else:
        return int(np.prod(shape));
