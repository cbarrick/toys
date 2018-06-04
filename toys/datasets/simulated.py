from typing import Any, Sequence

import numpy as np

import toys
from toys.common import Dataset


class SimulatedPolynomial(Dataset):
    '''A simulated polynomial dataset with gaussian noise.

    A simulated polynomial dataset has two columns ``(input, target)``.
    Both columns may have arbitrary shape.

    The inputs, true weights, and true bias are drawn from a standard uniform
    distribution. The noise is drawn from a standard normal distribution.
    '''

    def __init__(self, length, degree, *, in_shape=5, out_shape=3,
            bias=True, noise=True, seed='train'):
        '''Initialize a simulated polynomial dataset.

        Arguments:
            length (int):
                The number of datapoints.
            degree (int):
                The degree of the signal.

        Keyword Arguments:
            in_shape (int or Sequence[int]):
                The shape of data in the input column.
            out_shape (int or Sequence[int]):
                The shape of data in the target column.
            bias (bool):
                If true, apply a constant offset to the targets.
            noise (bool):
                If true, apply gaussian noise to the targets.
            seed (Any):
                A seed for the random number generator.
                Prefer strings like 'train' and 'test'.
        '''
        # The true signal must always be the same for every instance.
        rng = np.random.RandomState(0xDEADBEEF)
        weight_shape = (np.prod(in_shape), np.prod(out_shape))
        self.weights = [rng.uniform(size=weight_shape) for _ in range(degree)]
        self.bias = rng.uniform(size=(out_shape)) if bias else 0

        self.length = length
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.noise = bool(noise)
        self.seed = abs(hash(seed)) % (2 ** 32)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        rng = np.random.RandomState(self.seed + index)
        noise = rng.normal(size=self.out_shape) if self.noise else 0
        x = rng.uniform(size=self.in_shape).flatten()
        y = sum(x**(i+1) @ weight for i, weight in enumerate(self.weights))
        y += self.bias + noise
        return x, y

    @property
    def hints(self):
        return {
            'shuffle': False,
            'batch_size': 256,
        }


class SimulatedLinear(SimulatedPolynomial):
    '''A simulated linear dataset with gaussian noise.

    This is simply sugar for `SimulatedPolynomial` with degree 1.
    '''

    def __init__(self, length, **kwargs):
        '''Initialize a simulated linear dataset.

        See `SimulatedPolynomial` for a description of the arguments.
        '''
        super().__init__(length, degree=1, **kwargs)
