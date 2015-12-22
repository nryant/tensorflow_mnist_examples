"""Learning/momentum schedule classes.

The decision as to what schedules and hyperparameter settings to use for
the network learning rate and momentum parameters is nontrivial. For an
overview of the empirical peformance of different schedules (as well as
methods like AdaGrad) for speech, consult Senior et al. (2013).


References
----------
Senior, Heigold, Ranzato, and Yang. 2013. "An empirical study of
learning rates in deep neural networks for speech recognition."
*ICASSP*. 2013.
"""
from functools import wraps;

__all__ = ['ConstantSchedule', 'LinearSchedule', 'ExponentialSchedule',
           'PowerSchedule'];


class Schedule(object):
    """Schedule class template.

    Schedule class instances are used to determine the appropriate value
    for learning rate, momentum, or other parameter as a function f(t)
    of how many training instances have been seen so far.
    """
    def __call__(self, t):
        """Calculate some function f(t) of t, the number of training
        instances seen so far.

        Parameters
        ----------
        t :  int
            Number of epochs seen.

        Returns
        -------
        val : float
            Appropiate parameter value for this epoch under schedule.
        """
        return self.eta;


class ConstantSchedule(Schedule):
    """A schedule that returns the same value at every point in time:
    f(t) = eta.

    Parameters
    ----------
    eta : float
        Constant value for assignment.
    """
    def __init__(self, eta):
        self.eta = eta;

    @wraps(Schedule.__call__)
    def __call__(self, t):
        return self.eta;


class LinearSchedule(Schedule):
    """A schedule under which the parameter value after t instances is
    given by f(t) = eta0 + m*t.

    Parameters
    ----------
    eta0 : float
        Initial parameter value.

    m : float
        Slope of linear function.

    eta_min : float, optional
        Lower bound for parameter value.
        (Default: Values are unbounded.)

    eta_max : float, optional
        Upper bound for parameter value.
        (Default: Values are unbounded.)
    """
    def __init__(self, eta0, m, eta_min=None, eta_max=None):
        self.eta0 = eta0;
        self.m = m;
        self.eta_min = eta_min;
        self.eta_max = eta_max;

    @wraps(Schedule.__call__)
    def __call__(self, t):
        eta = self.eta0 + self.m*t;
        if not self.eta_min is None:
            eta = max(self.eta_min, eta);
        if not self.eta_max is None:
            eta = min(self.eta_max, eta);
        return eta;


class ExponentialSchedule(Schedule):
    """A schedule under which the parameter value after t instances is
    given by f(t) = eta0*base^(-t/r)

    Parameters
    ----------
    eta0 : float
        Initial parameter value.

    r : float
        TODO.

    base : float, optional
        TODO.
        (Default: 10)
    """
    def __init__(self, eta0, r, base=10.):
        self.eta0 = eta0;
        self.r = r;
        self.base = base;

    @wraps(Schedule.__call__)
    def __call__(self, t):
        return self.eta0*self.base**(-t/float(self.r));


class PowerSchedule(Schedule):
    """A schedule under which the parameter value after t instances is
    given by f(t) = eta0*(1 + t/r)^-c.

    Parameters
    ----------
    eta0 : float
        Initial parameter value.

    r : float
        TODO.

    c : float, optional
        "Problem independent constant." Bottou uses 1 for SGD and 0.75
        for Averaged SGD.
        (Default: 1)
    """
    def __init__(self, eta0, r, c=1.):
        self.eta0 = eta0;
        self.r = r;
        self.c = c;

    @wraps(Schedule.__call__)
    def __call__(self, t):
        return self.eta0*(1 + t/float(self.r))**-self.c;
