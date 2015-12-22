"""TODO
"""
import gzip;
import os;
import urllib;

import numpy as np;

__all__ = ['get_mnist_data'];


def get_mnist_data(mnist_dir, flatten_images=True, one_hot=True,
                   dtype='float32'):
    """Load MNIST images and labels as NumPy arrays.

    If the images/labels files do not already exist in ``mnist_dir``,
    then they will be downloaded.

    Parameters
    ----------
    mnist_dir : str
        Directory to search for images/labels files.

    flatten_images : bool, optional
        If True, flatten images into 784-D vectors. Otherwise, return
        each image as a 28 x 28 x 1 array (height x width x n_channels).
        (Default: True)

    one_hot : bool, optional
       If True, use one-hot coding for labels. Otherwise, represent
       labels as a 1-D array of integers.
       (Default: True)

    dtype : str or numpy.dtype, optional
        Datatype for returned arrays.
        (Default: 'float32')

    Returns
    -------
    X_tr : ndarray
        Training images. If ``flatten_images=True``, then a 60k x 784
        array. If ``flatten_images=False``, then a 60k x 28 x 28 x 1
        array in which each image is represented as a 28 pixel x 28
        pixel x 1 channel array.

    Y_tr : ndarray
        Training labels. If ``one_hot=True``, then a 60k x 10 array,
        each row of which is a one-hot vector encoding a label. If
        ``one_hot=False`` then a 60k x 1 array, each entry of which is
        an integer label.

    X_tst : ndarray
        Test images. If ``flatten_images=True``, then a 10k x 784 array.
        If ``flatten_images=False``, then a 10k x 28 x 28 x 1 array in
        which each image is represented as a 28 pixel x 28 pixel x 1
        channel array.

    Y_tst : ndarray
        Test labels. If ``one_hot=True``, then a 10k x 10 array, each
        row of which is a one-hot vector encoding a label. If
        ``one_hot=False`` then a 10k x 1 array, each entry of which is
        an integer label.
    """
    if not os.path.exists(mnist_dir):
        os.mkdir(mnist_dir);

    def download(fn):
        source = 'http://yann.lecun.com/exdb/mnist/';
        bn = os.path.basename(fn);
        urllib.urlretrieve(source + bn, fn);

    def load_images(fn):
        if not os.path.exists(fn):
            download(fn);

        # Convert to numpy array of desired dtype, rescale to the
        # interval [0, 1], and (if specified) reshape into 28 x 28
        # pixel single channel images.
        with gzip.open(fn, 'rb') as f:
            X = np.frombuffer(f.read(), np.uint8, offset=16);
        X = X.astype(dtype) / 255.;
        if not flatten_images:
            X = X.reshape(-1, 28, 28, 1);
        else:
            X = X.reshape(-1, 784);

        return X;

    def load_targets(fn):
        if not os.path.exists(fn):
            download(fn);

        # Convert to numpy array of desired dtype. Optionally, convert
        # to one-hot.
        with gzip.open(fn, 'rb') as f:
            Y = np.frombuffer(f.read(), np.uint8, offset=8);
        if one_hot:
            Y = np.column_stack([Y==ii for ii in range(10)]);
        Y = Y.astype(dtype);

        return Y;

    X_tr = load_images(os.path.join(mnist_dir,
                                    'train-images-idx3-ubyte.gz'));
    Y_tr = load_targets(os.path.join(mnist_dir,
                                     'train-labels-idx1-ubyte.gz'));
    X_tst = load_images(os.path.join(mnist_dir,
                                     't10k-images-idx3-ubyte.gz'));
    Y_tst = load_targets(os.path.join(mnist_dir,
                                      't10k-labels-idx1-ubyte.gz'));

    return X_tr, Y_tr, X_tst, Y_tst;
