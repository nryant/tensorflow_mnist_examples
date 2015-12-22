#!/usr/bin/env python

import sys;
import time;

import numpy as np;
import tensorflow as tf;

from mnist_lib.datasets import get_mnist_data, Dataset;
from mnist_lib.models import FullyConnectedModel;
from mnist_lib.schedules import *;
from mnist_lib.train import train_sgd;


MNIST_DIR = '../mnist';

# Network vars.
N_HID = [3];
HID_SIZE = [512, 1024, 2048];
INPUT_DROPOUT = 0.2;
HID_DROPOUT = 0.3;

# Training vars.
MBSZ = [128, 256, 512, 1024];
EPOCH_SIZE = 6e4;
N_EPOCHS = 10;
CLIP_VAL = 3.;
INIT_LEARN_RATE = 0.1;
INIT_MOMENTUM = 0.5;


def get_timing(n_hid, hid_size, mbsz, X_tr, Y_tr):
    data = Dataset(X_tr, Y_tr, mbsz=mbsz, method='sequential');
    n_samp, n_feats = X_tr.shape;
    n_targets = Y_tr.shape[1];
    model = FullyConnectedModel(n_feats, n_targets, n_hid,
                                hid_size, 'relu', None,
                                INPUT_DROPOUT, HID_DROPOUT);
    learn_rate_schedule = ExponentialSchedule(INIT_LEARN_RATE, 500);
    momentum_schedule = LinearSchedule(INIT_MOMENTUM, 0.001,
                                       eta_max=0.99);
    t1 = time.time();
    for ep, loss in train_sgd(model, data, N_EPOCHS, EPOCH_SIZE,
                              learn_rate_schedule, momentum_schedule,
                              CLIP_VAL):
        pass;
    t2 = time.time();
    return (t2 - t1) / float(N_EPOCHS);


if __name__ == '__main__':
    logf = sys.argv[1];

    with open(logf, 'wb') as f:
        X_tr, Y_tr, X_tst, Y_tst = get_mnist_data(MNIST_DIR);
        for n_hid in N_HID:
            for hid_size in HID_SIZE:
                for mbsz in MBSZ:
                    g = tf.Graph();
                    config = tf.ConfigProto(log_device_placement=True);
                    with tf.Session(graph=g, config=config) as sess:
                        t = get_timing(n_hid, hid_size, mbsz, X_tr,
                                       Y_tr);
                        fmt_str = 'h_hid: %d, hid_size: %d, mbsz: %d, per_epoch: %f\n';
                        f.write(fmt_str % (n_hid, hid_size, mbsz, t));
                        f.flush();
