from sys import float_info

import numpy as np

from .core import Accumulator, Mean, Sum


# Use epsilon only to prevent ZeroDivisionError.
# Rounding error may exceed epsilon.
EPSILON = float_info.epsilon


class Accuracy(Accumulator):
    supervised = True

    def __init__(self, **kwargs):
        self.val = Mean(**kwargs)

    def accumulate(self, y, h):
        y = np.asarray(y)
        h = np.asarray(h)
        val = (y == h)
        self.val.accumulate(val)

    def reduce(self):
        return self.val.reduce()


class TruePositives(Accumulator):
    supervised = True

    def __init__(self, target=1, **kwargs):
        self.target = target
        self.val = Sum(**kwargs)

    def accumulate(self, y, h):
        y = np.asarray(y)
        h = np.asarray(h)
        val = (h == self.target) & (y == self.target)
        self.val.accumulate(val)

    def reduce(self):
        return self.val.reduce()


class FalsePositives(Accumulator):
    supervised = True

    def __init__(self, target=1, **kwargs):
        self.target = target
        self.val = Sum(**kwargs)

    def accumulate(self, y, h):
        y = np.asarray(y)
        h = np.asarray(h)
        val = (h == self.target) & (y != self.target)
        self.val.accumulate(val)

    def reduce(self):
        return self.val.reduce()


class TrueNegatives(Accumulator):
    supervised = True

    def __init__(self, target=1, **kwargs):
        self.target = target
        self.val = Sum(**kwargs)

    def accumulate(self, y, h):
        y = np.asarray(y)
        h = np.asarray(h)
        val = (h != self.target) & (y != self.target)
        self.val.accumulate(val)

    def reduce(self):
        return self.val.reduce()


class FalseNegatives(Accumulator):
    supervised = True

    def __init__(self, target=1, **kwargs):
        self.target = target
        self.val = Sum(**kwargs)

    def accumulate(self, y, h):
        y = np.asarray(y)
        h = np.asarray(h)
        val = (h != self.target) & (y == self.target)
        self.val.accumulate(val)

    def reduce(self):
        return self.val.reduce()


class Precision(Accumulator):
    supervised = True

    def __init__(self, target=1, **kwargs):
        self.tp = TruePositives(target, **kwargs)
        self.fp = FalsePositives(target, **kwargs)

    def accumulate(self, y, h):
        y = np.asarray(y)
        h = np.asarray(h)
        self.tp.accumulate(y, h)
        self.fp.accumulate(y, h)

    def reduce(self):
        tp = self.tp.reduce()
        fp = self.fp.reduce()
        return tp / (tp + fp + EPSILON)


class Recall(Accumulator):
    supervised = True

    def __init__(self, target=1, **kwargs):
        self.tp = TruePositives(target, **kwargs)
        self.fn = FalseNegatives(target, **kwargs)

    def accumulate(self, y, h):
        y = np.asarray(y)
        h = np.asarray(h)
        self.tp.accumulate(y, h)
        self.fn.accumulate(y, h)

    def reduce(self):
        tp = self.tp.reduce()
        fn = self.fn.reduce()
        return tp / (tp + fn + EPSILON)


class FScore(Accumulator):
    supervised = True

    def __init__(self, beta=1, target=1, **kwargs):
        self.beta = beta
        self.tp = TruePositives(target, **kwargs)
        self.fp = FalsePositives(target, **kwargs)
        self.fn = FalseNegatives(target, **kwargs)

    def accumulate(self, y, h):
        y = np.asarray(y)
        h = np.asarray(h)
        self.tp.accumulate(y, h)
        self.fp.accumulate(y, h)
        self.fn.accumulate(y, h)

    def reduce(self):
        tp = self.tp.reduce()
        fp = self.fp.reduce()
        fn = self.fn.reduce()
        beta2 = self.beta ** 2
        tp2 = (1 + beta2) * tp
        fn2 = beta2 * fn
        return tp2 / (tp2 + fn2 + fp + EPSILON)


class MeanSquaredError(Accumulator):
    supervised = True

    def __init__(self, **kwargs):
        self.mean = Mean(**kwargs)

    def accumulate(self, y, h):
        y = np.asarray(y)
        h = np.asarray(h)
        err = (y - h) ** 2
        self.mean.accumulate(err)

    def reduce(self):
        return self.mean.reduce()


class NegMeanSquaredError(Accumulator):
    supervised = True

    def __init__(self, **kwargs):
        self.mean = Mean(**kwargs)

    def accumulate(self, y, h):
        y = np.asarray(y)
        h = np.asarray(h)
        err = (y - h) ** 2
        self.mean.accumulate(err)

    def reduce(self):
        return -self.mean.reduce()
