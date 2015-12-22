#!/usr/bin/env python

import cPickle;
import time;

import numpy as np;
import tensorflow as tf;

from mnist_lib.datasets import get_mnist_data, Dataset;
from mnist_lib.models import FullyConnectedModel;
from mnist_lib.schedules import *;
from mnist_lib.train import train_adam;

MNIST_DIR = '../mnist';

# Network vars.
N_HID = 2;
HID_SIZE = 800;
INPUT_DROPOUT = 0.2;
HID_DROPOUT = 0.3;

# Training vars.
MBSZ = 512;
EPOCH_SIZE = 6e4;
N_EPOCHS = 2000;
CLIP_VAL = 3.;
INIT_LEARN_RATE = 0.001;


if __name__ == '__main__':
    print('Generating feats/targets...');
    mnist_dir = '/data1/mnist';
    X_tr, Y_tr, X_tst, Y_tst = get_mnist_data(mnist_dir);
    n_samp, n_feats = X_tr.shape;
    n_targets = Y_tr.shape[1];

    print('Setting up session...');
    sess = tf.InteractiveSession();

    print('Setting up network...');
    with tf.variable_scope('') as vs:
        train_model = FullyConnectedModel(n_feats, n_targets, N_HID,
                                          HID_SIZE, 'relu', None,
                                          INPUT_DROPOUT, HID_DROPOUT);
        vs.reuse_variables();
        test_model = FullyConnectedModel(n_feats, n_targets, N_HID,
                                         HID_SIZE, 'relu', 'softmax',
                                         0., 0.);

    print('Training...');
    learn_rate_schedule = ConstantSchedule(INIT_LEARN_RATE);
    data = Dataset(X_tr, Y_tr, mbsz=MBSZ, method='sequential');
    t1 = time.time();
    for ep, loss in train_adam(train_model, data, N_EPOCHS, EPOCH_SIZE,
                              learn_rate_schedule, CLIP_VAL):
        t = time.time() - t1;
        if ep % 10 == 0:
            preds = test_model.fprop(X_tst);
            error = np.mean(preds.argmax(axis=1) != Y_tst.argmax(axis=1));
            print('Epoch %d: %f, %f, %f' % (ep, loss, t, error));
        else:
            print('Epoch %d: %f, %f' % (ep, loss, t));

    t2 = time.time();
    print (t2 - t1), (t2-t1)/float(N_EPOCHS);

    print('Saving model...');
    test_model.save_weights('models/adam.vars')
    with open('models/adam.pkl', 'w') as f:
        cPickle.dump(test_model, f, protocol=2);
