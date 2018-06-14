import numpy as np

from .core import Accumulator, Mean


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


class NegMeanSquaredError(MeanSquaredError):
    supervised = True

    def reduce(self):
        return -super().reduce()
