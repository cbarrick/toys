from typing import Any

import numpy as np

import toys
from toys.datasets.utils import Dataset


class SimulatedLinear(Dataset):
    '''A simulated linear dataset with gaussian noise.

    The inputs are drawn from a uniform distribution over [0,1). The true
    weights are sequential integers over [0, in_features). The true bias are
    sequential integers over [0, out_features). The noise is drawn from a
    standard normal distribution.
    '''

    def __init__(self, length, in_features=5, out_features=3, noise=True, seed='train'):
        '''Initialize a simulated dataset.

        Arguments:
            length (int):
                The number of datapoints.
            in_features (int):
                The number of features in the input data.
            out_features (int):
                The number of features in the targets.
            noise (bool):
                Set false to disable noise.
            seed (Any):
                A seed for the random number generator. Prefer to use strings
                like 'train' and 'test'.
        '''
        seed = abs(hash(seed)) % (2 ** 32)

        weight = np.arange(in_features * out_features)
        weight = weight.reshape(in_features, out_features)
        bias = np.arange(out_features)
        self.weight = weight
        self.bias = bias

        self.in_features = in_features
        self.out_features = out_features
        self.length = length
        self.noise = bool(noise)
        self.seed = seed

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        rng = np.random.RandomState(self.seed + index)
        error = rng.normal(size=self.out_features) if self.noise else 0
        x = rng.uniform(size=self.in_features)
        y = x @ self.weight + self.bias + error
        return x, y
