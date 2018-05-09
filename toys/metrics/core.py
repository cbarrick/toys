from abc import abstractmethod
from typing import Callable

import toys
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
    def __init__(self):
        self.val = 0

    def accumulate(self, batch):
        try:
            self.val += batch.sum(axis=0)
        except AttributeError:
            self.val += sum(batch)

    def reduce(self):
        val = self.val
        self.val = 0
        return val


class Mean(Accumulator):
    def __init__(self, fn=None):
        self.fn = fn
        self.n = 0
        self.val = 0

    def accumulate(self, batch):
        if self.fn is not None:
            batch = self.fn(batch)

        n = len(batch)

        try:
            val = batch.mean(axis=0)
        except (AttributeError, RuntimeError):
            # Torch throws RuntimeError when tensors do not implement `mean`.
            # WARNING: The naive mean formula is unstable. The assumption is
            # that the batch is small enough to avoid stability issues.
            val = sum(batch) / n

        # Update the global mean with Chan's algorithm, which is stable:
        # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
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
