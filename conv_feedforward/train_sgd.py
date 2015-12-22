#!/usr/bin/env python

import cPickle;
import time;

import numpy as np;
import tensorflow as tf;

from mnist_lib.datasets import get_mnist_data, Dataset;
from mnist_lib.models import ConvModel;
from mnist_lib.schedules import *;
from mnist_lib.train import train_sgd;

MNIST_DIR = '../mnist';

# Network vars.
N_FC = 2;
FC_SIZE = 256;
DROPOUT = 0.5;

# Training vars.
MBSZ = 512;
EPOCH_SIZE = 6e4;
N_EPOCHS = 2000;
CLIP_VAL = 2.;
INIT_LEARN_RATE = 0.01;
INIT_MOMENTUM = 0.9;


if __name__ == '__main__':
    print('Generating feats/targets...');
    X_tr, Y_tr, X_tst, Y_tst = get_mnist_data(MNIST_DIR,
                                              flatten_images=False);
    image_shape = X_tr.shape[1:];
    n_targets = Y_tr.shape[1];

    # Set up session for training.
    print('Setting up session...');
    sess = tf.InteractiveSession();

    # Set up network.
    print('Setting up network...');
    learn_rate_schedule = ConstantSchedule(INIT_LEARN_RATE);
    momentum_schedule = ConstantSchedule(INIT_MOMENTUM);
    with tf.variable_scope('') as vs:
        train_model = ConvModel(image_shape, n_targets, N_FC, FC_SIZE,
                                'relu', None, DROPOUT);
        vs.reuse_variables();
        test_model = ConvModel(image_shape, n_targets, N_FC, FC_SIZE,
                               'relu', 'softmax', 0.0);

    # Train.
    print('Training...');
    data = Dataset(X_tr, Y_tr, mbsz=MBSZ, method='sequential');
    t1 = time.time();
    for ep, loss in train_sgd(train_model, data, N_EPOCHS, EPOCH_SIZE,
                              learn_rate_schedule, momentum_schedule,
                              CLIP_VAL):
        t = time.time() - t1;
        if ep % 1 == 0:
            preds = test_model.fprop(X_tst);
            error = np.mean(preds.argmax(axis=1) != Y_tst.argmax(axis=1));
            print('Epoch %d: %f, %f, %f' % (ep, loss, t, error));
        else:
            print('Epoch %d: %f, %f' % (ep, loss, t));

    t2 = time.time();
    print (t2 - t1), (t2-t1)/float(N_EPOCHS);
