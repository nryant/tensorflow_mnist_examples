"""Classes and functions for generating minibatches of examples for
training.
"""
import numpy as np;

__all__ = ['Dataset'];


_METHODS = set(['sequential',
                'random_slices',
                ]);

class Dataset(object):
    """Iterator over minibatches of examples.

    Parameters
    ----------
    feats : ndarray, shape (n_examples, ...)
        Features.

    targets : ndarray, shape (n_examples,)
        Targets.

    n_classes : int, optional
        Number of classes in the classifications problem, which is used
        to code the label ``targets[i]`` with a 1-of-k code. If None,
        ``targets`` is treated as numeric.
        (Default: None)

    mbsz : int, optional
        Number of examples per minibatch.
        (Default: 128)

    method : str, optional
        Method used for iterating over dataset. If ``method=sequential``
        contiguous non-overlapping minibatches are generated
        sequentially. If ``method=random_slices``, contiguous slices
        with random start indices are generated.
        (Default: 'random_slices')

    shuffle : bool, optional
        If True, examples are shuffled in place before the first
        iteration.
        (Default: True)

    seed : int, optional
        Seed for random number generator. If None, then ``seed`` is
        generated by reading /dev/urandom (or Windows equivalent) if
        available or from clock otherwise.
        (Default: None)

    Attributes
    ----------
    n_examples : int
        Number of examples.

    rng : np.random.RandomState
        Random number generator.

    mb_gen: generator
        Actual generator that yields the minibatches.
    """
    def __init__(self, feats, targets, n_classes=None, mbsz=128,
                 method='random_slices', shuffle=True, seed=None):
        self.__dict__.update(locals());
        del self.self;

        # Convert features/targets to desired float type.
        self.feats = feats;
        if not n_classes is None:
            # If target is a 1-D array of integer labels AND we know the
            # number of classes, convert to 1-of-k coding.
            targets = np.column_stack([targets==ii
                                       for ii in xrange(n_classes)]);
        if not targets is feats:
            targets = targets.astype(self.feats.dtype, copy=False);
        self.targets = targets;
        self.n_examples = self.feats.shape[0];

        # Initialize RNG and perform initial shuffle of examples.
        self.rng = np.random.RandomState(seed);
        if shuffle:
            self.shuffle_data();

        # Initialize actual minibatch generator.
        assert(method in _METHODS);
        if method == 'sequential':
            def gen():
                slices = [];
                for ii in xrange(0, self.n_examples, self.mbsz):
                    slices.append(slice(ii, ii + self.mbsz));
                while True:
                    for slice_ in slices:
                        yield feats[slice_], targets[slice_];
        elif method == 'random_slices':
            def gen():
                max_bi = self.n_examples - self.mbsz;
                while True:
                    bi = self.rng.randint(max_bi);
                    ei = bi + mbsz;
                    yield self.feats[bi:ei, ], self.targets[bi:ei, ];
        self.mb_gen = gen();

    def shuffle_data(self):
        """
        """
        orig_state = self.rng.get_state();
        self.rng.shuffle(self.feats);
        self.rng.set_state(orig_state);
        self.rng.shuffle(self.targets);

    def next(self):
        """
        """
        return self.mb_gen.next();