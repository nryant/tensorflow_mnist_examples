"""Basic feedforward networks.
"""
from functools import partial, wraps;
import yaml;

import numpy as np;
import tensorflow as tf;

from .utils import get_randn_variable, get_zeros_variable, init_uninit;

__all__ = ['FullyConnectedModel', 'Model']


_IDENT_NAMES = set(['ident', 'identity', 'linear']);

def get_activation(activation, **kwargs):
    """Return tensorflow activation function from name and keyword
    arguments.
    """
    if activation in _IDENT_NAMES or activation is None:
        activation_func = tf.identity;
    else:
        activation_func =  getattr(tf.nn, activation);
    f = partial(activation_func, **kwargs);
    return f;


class Model(object):
    """Basic model class that wraps a TensorFlow network graph.

    Attributes
    ----------
    x : Tensor
        Input placeholder.

    y : Tensor
        Output placeholder.

    variables : list of tensorflow.Variable
        Trainable variables (weights and biases).

    dtype : str or tensorflow or NumPy dtype
        The dtype for tensors in network.

    input_shape
    output_shape
    """
    dtype = tf.float32;

    def __init__(self):
        raise NotImplementedError;

    def init_variables(self):
        """Initialize all variables unless inside a variable scope where
        ``reuse=True``.
        """
        if not tf.get_variable_scope().reuse:
            init = tf.initialize_variables(self.variables);
            init.run();

    def get_config(self):
        """Return dictionary of keyword arguments that when passed to
        ``__init__`` will reconstruct the network (modulo values of
        variables).
        """
        raise NotImplementedError;

    def fprop(self, x):
        """Propagate input forward through network.

        Parameters
        ----------
        x : ndarray or Tensor, (n_inputs, n_feats)
            Batch of inputs to network.

        Returns
        -------
        y : ndarray, (n_inputs, n_classes)
            Batch of outputs of final layer of network.
        """
        return self.y.eval(feed_dict={self.x : x});

    def predict(self, x):
        """Propagate input forward through network to make predictions.

        Parameters
        ----------
        x : ndarray or Tensor, (n_inputs, ...
            Batch of inputs to network.

        Returns
        -------
        y : ndarray, (n_inputs,)
            Batch of predictions. A vector of class integer labels.
        """
        return self.fprop(x).argmax(axis=1);

    def save_weights(self, fn):
        saver = tf.train.Saver(self.variables);
        saver.save(tf.get_default_session(), fn);

    def load_weights(self, fn):
        saver = tf.train.Saver(self.variables);
        saver.restore(tf.get_default_session(), fn);

    def __getstate__(self):
        new_dict = self.__dict__.copy();
        new_dict['x'] = None;
        new_dict['y'] = None;
        new_dict['variables'] = None;
        return new_dict;

    def __setstate__(self, dict):
        self.__dict__.update(dict);
        self.__init__(**self.get_config());

    @property
    def input_shape(self):
        return self.x.get_shape();

    @property
    def output_shape(self):
        return self.y.get_shape();


class FullyConnectedModel(Model):
    """Simple fully connected feedforward neural network.

    Parameters
    ----------
    input_dim : int
        Dimensionality of input features.

    out_dim : int
        Dimensionality of output.

    n_hid : int
        Number of hidden layers.

    hid_dim : int
        Number of units in each hidden layer.

    hid_activation : str, optional
        Name of activation function to use for hidden layers. See also
        ``get_activation``.
        (Default: 'relu')

    out_activation : str, optional
        Name of activation function to use for output layer. See also
        ``get_activation``.

    input_dropout : float, optional
        Proportion of input units to dropout.
        (Default: 0.0)

    hidden_dropout : float, optional
        Proportion of output units to dropout.
        (Default: 0.0)
    """
    def __init__(self, input_dim, out_dim, n_hid, hid_dim,
                 hid_activation='relu', out_activation='softmax',
                 input_dropout=0.0, hidden_dropout=0.0):
        self.__dict__.update(locals());
        del self.self;

        # Determine layer sizes.
        layer_sizes = [input_dim] + [hid_dim]*n_hid + [out_dim];
        n_layers = len(layer_sizes);

        # Build input layer
        with tf.variable_scope('input') as vs:
            input_place = tf.placeholder('float32', [None, input_dim],
                                         name='act');
            acts = [input_place];
            if input_dropout > 0:
                acts.append(tf.nn.dropout(acts[-1], 1 - input_dropout,
                                          name='drop'));

        # Build hidden layers/output layer.
        variables = [];
        for ii, (prev_size, curr_size) in enumerate(zip(layer_sizes[:-1],
                                                        layer_sizes[1:])):
            is_hidden_layer = ii < len(layer_sizes) - 2;
            layer_name = 'hid_%d' % ii if is_hidden_layer else 'final';
            with tf.variable_scope(layer_name) as vs:
                # Initialize weights/biases for affine component.
                sigma = np.sqrt(2 / float(prev_size + curr_size));
                W = get_randn_variable('W', [prev_size, curr_size],
                                       tf.float32, 0, sigma);
                B = get_zeros_variable('B', [curr_size],
                                       tf.float32);
                variables.extend([W, B]);

                # Add affine component and following nonlinearity.
                if is_hidden_layer:
                    f = get_activation(hid_activation,
                                       name='act');
                else:
                    f = get_activation(out_activation, name='act');
                acts.append(f(tf.matmul(acts[-1], W) + B));

                # Apply dropout mask if requested.
                if is_hidden_layer and hidden_dropout > 0:
                    acts.append(tf.nn.dropout(acts[-1],
                                              1 - input_dropout,
                                              name='drop'));

        self.variables = variables;
        self.x = input_place;
        self.y = acts[-1];

        self.init_variables();

    @wraps(Model.get_config)
    def get_config(self):
        kwargs = {'input_dim' : self.input_dim,
                  'out_dim' : self.out_dim,
                  'n_hid' : self.n_hid, 'hid_dim' : self.hid_dim,
                  'hid_activation' : self.hid_activation,
                  'out_activation' : self.out_activation,
                  'input_dropout' : self.input_dropout,
                  'hidden_dropout' : self.hidden_dropout,
                  };
        return kwargs;


class ConvModel(Model):
    """Simple convolutional feedforward neural network.

    Parameters
    ----------
    input_dim : tuple
        Shape of input images (height x width x n_channels).

    out_dim : int
        Dimensionality of output.

    n_fc : int, optional
        Number of fully-connected layers.
        (Default: 2)

    n_fc_hid : int, optional
       Number of units in each fully-connected layer.
       (Default: 256)

    fc_activation : str, optional
        Name of activation function to use for fully-connected layers.
        See also ``get_activation``.
        (Default: 'relu')

    out_activation : str, optional
        Name of activation function to use for output layer. See also
        ``get_activation``.

    dropout : float, optional
        Proportion of units from the topmost convolutional layer and
        hidden layers to dropout.
        (Default: 0.5)
    """
    def __init__(self, input_shape, out_dim, n_fc=2, n_fc_hid=256,
                 fc_activation='relu', out_activation='softmax',
                 dropout=0.5):
        # TODO: Work out a better naming scheme.
        self.__dict__.update(locals());
        del self.self;

        image_height, image_width, n_channels = input_shape;

        # Build input layer
        with tf.variable_scope('input') as vs:
            input_place = tf.placeholder('float32',
                                         [None] + list(input_shape),
                                         name='act');
            acts = [input_place];

        # First convolutional (+pooling) layer.
        variables = [];
        with tf.variable_scope('conv1') as vs:
            # Convolutional component.
            filter_width = 5;
            filter_height = 5;
            n_filters = 32;
            stride = 1;
            sigma = np.sqrt(2 / float(n_filters + (filter_width*filter_height)));
            kernel_shape = [filter_height, filter_width, n_channels,
                            n_filters];
            kernel = get_randn_variable('W', kernel_shape, tf.float32, 0,
                                        1e-4);
            conv = tf.nn.conv2d(input_place, kernel, [1, stride, stride, 1],
                                padding = 'SAME');
            B = get_zeros_variable('B', [n_filters], tf.float32);
            acts.append(tf.nn.relu(conv + B, name='conv'));
            variables.extend([kernel, B]);

            # Pooling component.
            pool_width = 2;
            pool_height = 2;
            stride = 2;
            pool = tf.nn.max_pool(acts[-1],
                                  ksize=[1, pool_height, pool_width, 1],
                                  strides=[1, stride, stride, 1],
                                  padding='SAME', name='pool');
            acts.append(pool);

            # Normalization component.
            norm = tf.nn.lrn(acts[-1], 4, bias=1.0, alpha=0.001 / 9.0,
                             beta=0.75,
                             name='norm');
            acts.append(norm);

        # Second convolutional (+pooling) layer.
        with tf.variable_scope('conv2') as vs:
            # Convolutional component.
            filter_width = 5;
            filter_height = 5;
            n_channels = n_filters;
            n_filters = 32;
            stride = 1;
            sigma = np.sqrt(2 / float(n_filters + (filter_width*filter_height)));
            kernel_shape = [filter_height, filter_width, n_channels,
                            n_filters];
            kernel = get_randn_variable('W', kernel_shape, tf.float32,
                                        0, 1e-4);
            conv = tf.nn.conv2d(acts[-1], kernel,
                                [1, stride, stride, 1],
                                padding= 'SAME');
            B = get_zeros_variable('B', [n_filters], tf.float32);
            acts.append(tf.nn.relu(conv + B, name='conv'));
            variables.extend([kernel, B]);

            # Pooling component.
            pool_width = 2;
            pool_height = 2;
            pool_shape = [1, pool_height, pool_width, 1];
            stride = 2;
            pool = tf.nn.max_pool(acts[-1],
                                  ksize=[1, pool_height, pool_width, 1],
                                  strides=[1, stride, stride, 1],
                                  padding='SAME', name='pool');
            acts.append(pool);

            # Normalization component.
            norm = tf.nn.lrn(acts[-1], 4, bias=1.0, alpha=0.001 / 9.0,
                             beta=0.75, name='norm');
            acts.append(norm);

            # Dropout
            if dropout > 0.0:
                acts.append(tf.nn.dropout(acts[-1], 0.5, name='drop'));

        # Flatten tensors so that for fully-connected layers so can
        # use efficient SGEMM calls.
        n_cols = int(np.prod(map(int, acts[-1].get_shape()[1:])));
        reshape = tf.reshape(acts[-1], [-1, n_cols], name='reshape');
        acts.append(reshape);

        # Fully connected layers.
        layer_sizes = [n_cols] + [n_fc_hid]*n_fc;
        for ii, (prev_size, curr_size) in enumerate(zip(layer_sizes[:-1],
                                                        layer_sizes[1:])):
            with tf.variable_scope('fc_%d' % ii) as vs:
                # Init weights/biases and add matmul op.
                sigma = np.sqrt(2 / float(prev_size + curr_size));
                W = get_randn_variable('W', [prev_size, curr_size],
                                       tf.float32, 0, sigma);
                B = get_zeros_variable('B', [curr_size], tf.float32);
                variables.extend([W, B]);
                net_input = tf.matmul(acts[-1], W) + B;
                acts.append(tf.nn.relu(net_input, name='act'));

                if dropout > 0:
                    acts.append(tf.nn.dropout(acts[-1], 0.5,
                                              name='drop'));

        # Output layer.
        with tf.variable_scope('final') as vs:
            f = get_activation(out_activation, name=vs.name);

            # Init weights/biases and add matmul op.
            prev_size = curr_size;
            curr_size = out_dim;
            sigma = np.sqrt(2 / float(prev_size + curr_size));
            W = get_randn_variable('W', [prev_size, curr_size],
                                   tf.float32, 0, sigma);
            B = get_zeros_variable('B', [curr_size], tf.float32);
            variables.extend([W, B]);
            net_input = tf.matmul(acts[-1], W) +B;
            acts.append(f(net_input, name='act'));

        self.variables = variables;
        self.x = input_place;
        self.y = acts[-1];

        self.init_variables();

    def get_config(self):
        kwargs = {'input_shape' : self.input_shape,
                  'out_dim' : self.out_dim,
                  'n_fc' : self.n_fc,
                  'n_fc_hid' : self.n_fc_hid,
                  'fc_activation' : self.fc_activation,
                  'out_activation' : self.out_activation,
                  'dropout' : self.dropout,
                  };
        return kwargs;
