from abc import abstractmethod
from typing import Callable

import toys
from toys.core import Model
from toys.datasets.utils import Dataset

try:
    from typing import Protocol
except:
    from abc import ABC as Protocol


class Accumulator(Protocol):
    @abstractmethod
    def accumulate(self, *values):
        '''Update the accumulator with new observations.
        '''
        raise NotImplementedError

    @abstractmethod
    def reduce(self):
        '''Read the value of the accumulator and reset to the initial state.
        '''
        raise NotImplementedError


class Sum(Accumulator):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.val = 0

    def accumulate(self, batch):
        try:
            self.val += batch.sum(**self.kwargs)
        except (TypeError, AttributeError):
            self.val += batch

    def reduce(self):
        val = self.val
        self.val = 0
        return val


class Mean(Accumulator):
    def __init__(self, fn=None, **kwargs):
        self.kwargs = kwargs
        self.fn = fn
        self.n = 0
        self.val = 0

    def accumulate(self, batch):
        if self.fn is not None:
            batch = self.fn(batch)

        if hasattr(batch, 'mean'):
            n = len(batch)
            val = batch.mean(**self.kwargs)
        elif hasattr(batch, 'double'):
            n = len(batch)
            val = batch.double().mean(**self.kwargs)
        else:
            n = 1
            val = batch

        if self.n == 0:
            self.n = n
            self.val = val

        else:
            delta = val - self.val
            self.n += n
            self.val += delta * n / self.n

    def reduce(self):
        val = self.val
        self.n = 0
        self.val = 0
        return val
