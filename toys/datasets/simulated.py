from typing import Any

import numpy as np

import toys
from toys.datasets.utils import Dataset


class SimulatedLinear(Dataset):
    '''A simulated linear dataset with gaussian noise.

    The inputs, true weights, and true bias are drawn from a standard uniform
    distribution. The noise is drawn from a standard normal distribution.
    '''

    def __init__(self, length, in_features=5, out_features=3, bias=True, noise=True, seed='train'):
        '''Initialize a simulated dataset.

        Arguments:
            length (int):
                The number of datapoints.
            in_features (int):
                The number of features in the input data.
            out_features (int):
                The number of features in the targets.
            bias (bool):
                If true, apply a constant offset to the data.
            noise (bool):
                Set false to disable noise.
            seed (Any):
                A seed for the random number generator.
                Prefer strings like 'train' and 'test'.
        '''
        # The true signal must always be the same for every instance.
        rng = np.random.RandomState(0xDEADBEEF)
        self.weight = rng.uniform(size=(in_features, out_features))
        self.bias = rng.uniform(size=(out_features)) if bias else 0

        self.in_features = in_features
        self.out_features = out_features
        self.length = length
        self.noise = bool(noise)
        self.seed = abs(hash(seed)) % (2 ** 32)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        rng = np.random.RandomState(self.seed + index)
        error = rng.normal(size=self.out_features) if self.noise else 0
        x = rng.uniform(size=self.in_features)
        y = x @ self.weight + self.bias + error
        return x, y
