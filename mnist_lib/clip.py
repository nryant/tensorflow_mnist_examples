"""Implements an op that removes -infs, infs, and nans from a tensor and replaces
them with 0
"""
import numpy as np;
import tensorflow as tf;

__all__ = ['remove_inf_nan', 'clip'];


LARGE_VAL = 1e7; # Some unreasonably large number that

def remove_inf_nan(x):
    """Replace nan, -inf, and inf values in tensor with 0.

    Primarily, this is useful for conditioning gradients where problems
    with precision issues may lead to ``-inf``, ``inf``, and ``nan`
    values in gradients that we need to eliminate. To an extent, this
    can be performed with the TensorFlow ``clip_by_value`` op, which
    when applied to a tensor ``x`` as in ``clip_by_value(x, clip_min,
    clip_max)`` will map ``-inf`` values to ``clip_min``, ``inf``
    values to ``clip_max``, and ``nan`` values to ``clip_max``. However,
    this is not quite what we want as the presence of one of these
    values in a tensor indicates the computation has gone awry
    somewhere, in which case we shouldn't make **ANY** adjustment to
    the corresponding parameter in the gradient update; hence, should
    set the gradient to 0.

    Parameters
    ----------
    x : Tensor
        TensorFlow tensor.

    dev : str, optional
        Device name or function.

    Returns
    -------
    out : Tensor
        Tensor equivlant to ``x`` but with ``-inf``, ``inf``, and
        ``nan`` values replaced by 0.
    """
    # TODO:
    #     -benchmark vs tf.clip_by_value call to see how much overhead
    #      this introduces
    min_val = tf.cast(-LARGE_VAL, dtype=x.dtype);
    max_val = tf.cast(LARGE_VAL, dtype=x.dtype);
    clipped = tf.clip_by_value(x, min_val, max_val);
    keep = tf.not_equal(tf.abs(clipped), max_val);
    zeroed = clipped*tf.cast(keep, clipped.dtype);
    return zeroed;


def clip(x, clip_value_min, clip_value_max, remove_inf_nan=True,
         name=None):
    """Clip tensor values to specified range.

    Safer and more efficient than applying ``remove_inf_nan`` then
    ``clip_by_value``.

    Parameters
    ----------
    x : Tensor
        TensorFlow tensor.

    clip_value_min : 0-D Tensor
        The minimum value to clip by.

    clip_value_max : 0-D Tensor
        The maximum value to clip by.

    remove_inf_nan : bool, optional
        If True, map ``-inf``, ``inf``, and ```nan`` values of ``x`` to
        0 prior to clipping. Else, ``inf`` and ``nan`` will be mapped to
        ``clip_value_max`` and ``-inf`` to ``clip_value_min``.
       (Default: True)

    name : str, optional
        A name for the operation.
        (Default: None)

    Returns
    -------
    out : Tensor
        Clipped tensor.
    """
    if remove_inf_nan:
        clipped = tf.clip_by_value(x, clip_value_min, clip_value_max);
        keep = tf.is_finite(x);
        y = tf.mul(clipped, tf.cast(keep, clipped.dtype), name=name);
    else:
        clipped = tf.clip_by_value(x, clip_value_min, clip_value_max,
                                   name=name);
        y = clipped;
    return y;
