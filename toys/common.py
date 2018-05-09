from abc import ABC, abstractmethod
from typing import Any, Callable, Tuple

import numpy as np

import torch
from torch.nn import Module

import toys
from toys.datasets.utils import Dataset


Model = Callable
Estimator = Callable


class BaseEstimator(ABC):
    '''A useful base class for estimators.

    An estimator is any callable that returns a model. The `BaseEstimator` base
    class provides a convenient API for implementing estimators.

    Instances of `BasesEstimator` follow the estimator protocol: they are
    functions that take a dataset to fit and keyword arguments for any
    hyperparameters. The constructor of a `BaseEstimator` accepts the very
    same keyword arguments. When a `BaseEstimator` is called directly (or via
    `__call__`), it delegates to `fit`, forwarding all keyword arguments as
    well as those passed to the constructor. In case of conflict, the arguments
    passed directly take priority.

    This upshot is that the constructor allows you to set the default
    hyperparameters.
    '''

    def __init__(self, **kwargs):
        '''Construct an estimator.

        Arguments:
            **kwargs (Any):
                Overrides the default keyword arguments.
        '''
        super().__init__()
        self._kwargs = kwargs

    def __call__(self, dataset, **kwargs):
        '''Construct a model, delegating to `fit`.

        Returns:
            model (Model):
                The model returned by `fit`.
        '''
        kwargs = {**self._kwargs, **kwargs}
        return self.fit(dataset, **kwargs)

    @abstractmethod
    def fit(self, dataset, **kwargs):
        '''Constructs a model.

        Subclasses must implement this method.

        The return value can be any callable, and is usually some learned
        function. Meta-estimators like `GridSearchCV` return other estimators.

        Arguments:
            dataset (Dataset):
                The dataset to fit.
            **kwargs:
                The hyperparameters to use while training the model.

        Returns:
            model (Model):
                Any arbitrary callable.
        '''
        raise NotImplementedError()


class TorchModel(Model):
    '''A wrapper around PyTorch modules.

    This wrapper extends `torch.nn.Module` to accept scalars, numpy arrays, and
    torch tensors as input and to return numpy arrays as output.

    A `TorchModel` is aware of the number of dimensions expected for each
    input. If an input has fewer dimensions, trivial axes are added.

    ..note:
        A `TorchModel` is NOT a `torch.nn.Module`. Backprop graphs are not
        created during prediction.

    Attributes:
        module (Module):
            The module being wrapped.
        dims (Tuple[int or None] or None):
            The number of dimensions required of each input. If present,
            the number and order of dimensions must match the number and
            order of inputs expected by the module. A value of ``None``
            means any shape is allowed for the corresponding input. If not
            present, the number and shape of inputs is unconstrained.
    '''

    def __init__(self, module, *dims):
        '''Construct a `TorchModel`.

        Arguments:
            module (Module):
                The module being wrapped.
            dims (int or None):
                The number of dimensions required of each input. If present,
                the number and order of dimensions must match the number and
                order of inputs expected by the module. A value of ``None``
                means any shape is allowed for the corresponding input. If not
                present, the number and shape of inputs is unconstrained.
        '''
        self.module = module
        self.dims = dims or None

    def __getattr__(self, name):
        '''Attribute access is delecated to the underlying module.
        '''
        return getattr(self.module, name)

    def __call__(self, *inputs):
        '''Evaluate the model on some inputs.
        '''
        with torch.no_grad():
            inputs = self._cast_inputs(*inputs)
            y = self.module(*inputs)
            y = y.numpy()
            return y

    def _cast_inputs(self, *inputs):
        '''Cast inputs to tensors of the expected dtype, device, and dimension.
        '''
        assert self.dims is None or len(inputs) == len(self.dims)
        dtype = self.module.dtype
        device = self.module.device

        for i, x in enumerate(inputs):
            if np.isscalar(x):
                x = np.array(x)

            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)

            x = x.to(device, dtype)

            if self.dims and self.dims[i]:
                assert x.dim() <= self.dims[i]
                for _ in range(x.dim(), self.dims[i]):
                    x = x.unsqueeze_(0)

            yield x
