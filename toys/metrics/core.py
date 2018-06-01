from abc import ABC, abstractmethod
from typing import Callable

import torch

import toys
from toys.datasets.utils import Dataset
from toys.parsers import parse_metric


Metric = Callable


class MultiMetric(Metric):
    def __init__(self, metrics, **kwargs):
        self.metrics = [parse_metric(m) for m in metrics]
        self.kwargs = kwargs

    def __call__(self, model, *inputs, **kwargs):
        kwargs = {**self.kwargs, **kwargs}
        scores = tuple(m(model, *inputs, **kwargs) for m in self.metrics)
        if len(scores) == 1:
            return scores[0]
        else:
            return scores


class Accumulator(ABC):
    supervised = False

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

    def __call__(self, model, *datasets, dry_run=False):
        dataset = toys.zip(*datasets)
        for batch in toys.batches(dataset):
            if self.supervised:
                *inputs, target = batch
                prediction = model(*inputs)
                self.accumulate(target, prediction)
            else:
                prediction = model(*batch)
                self.accumulate(prediction)
            if dry_run:
                break

        score = self.reduce()
        return score


class Sum(Accumulator):
    def __init__(self, fn=None, dim=None):
        self.fn = fn
        self.dim = dim
        self.val = 0

    def accumulate(self, batch):
        if self.fn is not None:
            batch = self.fn(batch)

        if not torch.is_tensor(batch):
            batch = torch.tensor(batch)

        if self.dim is None:
            self.val += batch.sum()
        else:
            self.val += batch.sum(dim=self.dim)

    def reduce(self):
        val = self.val
        self.val = 0
        return val


class Mean(Accumulator):
    def __init__(self, fn=None, dim=None):
        self.fn = fn
        self.dim = dim
        self.n = 0
        self.val = 0

    def accumulate(self, batch):
        if self.fn is not None:
            batch = self.fn(batch)

        if not torch.is_tensor(batch):
            batch = torch.tensor(batch)

        if self.dim is None:
            n = len(batch)
            val = batch.mean()
        else:
            n = len(batch)
            val = batch.mean(dim=self.dim)

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
