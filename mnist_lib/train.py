"""Functions for training networks.
"""
import tensorflow as tf;

__all__ = ['train_adam', 'train_rmsprop', 'train_sgd'];


# TODO: Refactor so that trainers are subclasses of a base Trainer
#       class like I've used in Theano based approaches. Too much
#       duplication of code/docstrings right now.

def train_sgd(model, data, n_epochs, epoch_size, learn_rate_schedule,
              momentum_schedule, clip_val, session=None):
    """Train network using minibatch gradient descent and momentum.

    Parameters
    ----------
    model : models.Model instance
        Network to train.

    data : datasets.Dataset
        Training dataset.

    n_epochs : int
        Number of epochs to train for.

    epoch_size : int
        Number of minibatches per epoch.

    learn_rate_schedule : schedules.Schedule
        Schedule instance used to determine learning rate at each epoch.

    momentum_schedule : schedules.Schedule
        Schedule instance used to determine momentum at each epoch.

    clip_val : float
        L2 norm clip value.

    session : tf.Session, optional
        Session to run in. If None runs in the default session.
        (Default: None)

    Yields
    ------
    loss : float
        Total loss on current epoch.
    """
    if session is None:
        session = tf.get_default_session();

    # Build train_step.
    y_place = tf.placeholder(model.dtype, shape=model.output_shape);
    ce_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(model.y,
                                                                    y_place));
    lr_place = tf.placeholder(model.dtype);
    momentum_place = tf.placeholder(model.dtype);
    optimizer = tf.train.MomentumOptimizer(lr_place, momentum_place);
    grads_vars = optimizer.compute_gradients(ce_loss);
    if clip_val is not None:
        grads_vars = [(tf.clip_by_norm(grad, clip_val), var) for grad, var in grads_vars];
    train_step = optimizer.apply_gradients(grads_vars);

    # Initialize variables used by optimizer.
    opt_vars = list(set(tf.all_variables()) -
                    set(tf.trainable_variables()));
    init = tf.initialize_variables(opt_vars);
    session.run(init);

    # Train.
    batches_per_epoch = int(epoch_size / data.mbsz);
    for ep in xrange(1, n_epochs+1):
        curr_learn_rate = learn_rate_schedule(ep-1);
        curr_momentum = momentum_schedule(ep-1);
        loss = 0;
        for ii in xrange(batches_per_epoch):
            x, y = next(data);
            feed_dict = {model.x : x,
                         y_place : y,
                         lr_place : curr_learn_rate,
                         momentum_place : curr_momentum,
            };
            _, curr_loss = session.run([train_step, ce_loss],
                                       feed_dict = feed_dict)
            loss += curr_loss;
        yield ep, loss;


def train_adam(model, data, n_epochs, epoch_size, learn_rate_schedule,
               clip_val, beta1=0.9, beta2=0.999, epsilon=1e-8,
               session=None):
    """Train network using ADAM.

    **NOTE** I have been unable to replicate the results on MNIST
    reported in the ADAM paper.

    Parameters
    ----------
    model : models.Model instance
        Network to train.

    data : datasets.Dataset
        Training dataset.

    n_epochs : int
        Number of epochs to train for.

    epoch_size : int
        Number of minibatches per epoch.

    learn_rate_schedule : schedules.Schedule
        Schedule instance used to determine learning rate at each epoch.

    clip_val : float
        L2 norm clip value.

    session : tf.Session, optional
        Session to run in. If None runs in the default session.
        (Default: None)

    Yields
    ------
    loss : float
        Total loss on current epoch.
    """
    if session is None:
        session = tf.get_default_session();

    # Build train_step.
    y_place = tf.placeholder(model.dtype, shape=model.output_shape);
    ce_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(model.y,
                                                                    y_place));
    lr_place = tf.placeholder(model.dtype);
    optimizer = tf.train.AdamOptimizer(lr_place, beta1, beta2, epsilon);
    grads_vars = optimizer.compute_gradients(ce_loss);
    if clip_val is not None:
        grads_vars = [(tf.clip_by_norm(grad, clip_val), var) for grad, var in grads_vars];
    train_step = optimizer.apply_gradients(grads_vars);

    # Initialize variables used by optimizer.
    opt_vars = list(set(tf.all_variables()) -
                    set(tf.trainable_variables()));
    for var in tf.all_variables():
        if var.name in ['beta1_power:0', 'beta2_power:0']:
            opt_vars.append(var);
    init = tf.initialize_variables(opt_vars);
    session.run(init);

    # Train.
    batches_per_epoch = int(epoch_size / data.mbsz);
    for ep in xrange(1, n_epochs+1):
        curr_learn_rate = learn_rate_schedule(ep-1);
        loss = 0;
        for ii in xrange(batches_per_epoch):
            x, y = next(data);
            feed_dict = {model.x : x,
                         y_place : y,
                         lr_place : curr_learn_rate,
            };
            _, curr_loss = session.run([train_step, ce_loss],
                                       feed_dict = feed_dict)
            loss += curr_loss;
        yield ep, loss;


def train_rmsprop(model, data, n_epochs, epoch_size,
                  learn_rate_schedule, momentum_schedule, clip_val,
                  decay=0.9, epsilon=1e-6, session=None):
    """

    Parameters
    ----------
    model : models.Model instance
        Network to train.

    data : datasets.Dataset
        Training dataset.

    n_epochs : int
        Number of epochs to train for.

    epoch_size : int
        Number of minibatches per epoch.

    learn_rate_schedule : schedules.Schedule
        Schedule instance used to determine learning rate at each epoch.

    momentum_schedule : schedules.Schedule
        Schedule instance used to determine momentum at each epoch.

    clip_val : float
        L2 norm clip value.

    decay
    epsilon

    session : tf.Session, optional
        Session to run in. If None runs in the default session.
        (Default: None)

    Yields
    ------
    loss : float
        Total loss on current epoch.
    """
    if session is None:
        session = tf.get_default_session();

    # Build train_step.
    y_place = tf.placeholder(model.dtype, shape=model.output_shape);
    ce_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(model.y,
                                                                    y_place));
    lr_place = tf.placeholder(model.dtype);
    momentum_place = tf.placeholder(model.dtype);
    optimizer = tf.train.RMSPropOptimizer(lr_place, decay,
                                          momentum_place, epsilon);
    grads_vars = optimizer.compute_gradients(ce_loss);
    if clip_val is not None:
        grads_vars = [(tf.clip_by_norm(grad, clip_val), var) for grad, var in grads_vars];
    train_step = optimizer.apply_gradients(grads_vars);

    # Initialize variables used by optimizer.
    opt_vars = list(set(tf.all_variables()) -
                    set(tf.trainable_variables()));
    init = tf.initialize_variables(opt_vars);
    session.run(init);

    # Train.
    batches_per_epoch = int(epoch_size / data.mbsz);
    for ep in xrange(1, n_epochs+1):
        curr_learn_rate = learn_rate_schedule(ep-1);
        curr_momentum = momentum_schedule(ep-1);
        loss = 0;
        for ii in xrange(batches_per_epoch):
            x, y = next(data);
            feed_dict = {model.x : x,
                         y_place : y,
                         lr_place : curr_learn_rate,
                         momentum_place : curr_momentum,
            };
            _, curr_loss = session.run([train_step, ce_loss],
                                       feed_dict = feed_dict)
            loss += curr_loss;
        yield ep, loss;
