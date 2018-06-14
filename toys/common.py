from abc import ABC, abstractmethod
from typing import *

import numpy as np
import pandas as pd

import torch
from torch.nn import Module, DataParallel


# The Protocol type does not exist until Python 3.7.
# TODO: Remove the try-except when Python 3.6 support is dropped.
try:
    from typing import Protocol
except ImportError:
    from abc import ABC as Protocol


class Estimator(Protocol):
    '''The estimator protocol.
    '''
    @abstractmethod
    def __call__(*args, **kwargs):
        raise NotImplementedError


class Model(Protocol):
    '''The model protocol.
    '''
    @abstractmethod
    def __call__(*args, **kwargs):
        raise NotImplementedError


# Common classes
# --------------------------------------------------

class BaseEstimator(ABC):
    '''A useful base class for estimators.

    An estimator is any callable that accepts zero or more inputs to be fit
    against, along with keyword arguments for any hyperparameters, and returns
    a model. This class provides a convenient API for estimators, allowing the
    default keyword arguments to be set by the constructor.

    When the estimator is invoked as a function, it delegates to the abstract
    method :meth:`fit`, taking any unset keyword arguments from those passed
    to the constructor. Subclasses then implement their estimator logic in
    :meth:`fit`.

    Attributes:
        defaults (Dict[str, Any]):
            The default kwargs for the instance.
    '''

    def __init__(self, **defaults):
        super().__init__()
        self.defaults = defaults

    def __call__(self, *args, **kwargs):
        kwargs = {**self.defaults, **kwargs}
        return self.fit(*args, **kwargs)

    @abstractmethod
    def fit(self, *args, **kwargs):
        '''Fit a model.

        Subclasses must implement this method.

        .. note::
            The recipe for fitting the model is defined by this method, but
            calling it directly circumvents the default keyword arguments set
            by the constructor. This is almost never desired. Always invoke
            the estimator instance rather than this method.

        Returns:
            Model:
                The fitted model.
        '''
        raise NotImplementedError()


class TorchModel(Module):
    '''A convenience wrapper around PyTorch modules.

    The model is distributed over all available GPUs by default.

    The model has a dtype specified during construction, and the underlying
    module and all inputs are automatically cast to this dtype. This largely
    removes the user from manual dtype casting.

    The model distinguishes between training and evaluation modes.
    In training/evaluation mode, ``autograd`` is enabled/disabled explicitly.
    This largely removes the user from manual autograd management. When
    a :class:`TorchModel` is constructed, it is set to training mode.
    Estimators should return models in evaluation mode.

    Arguments:
        module (Module):
            The module being wrapped.
        device_ids (Sequence[int]):
            A list of devices to use. The default is all available.
        dtype (str or torch.dtype):
            The dtype to which the module and inputs are cast.
    '''
    def __init__(self, module, device_ids=None, dtype='float32'):
        from toys.parsers import parse_dtype

        super().__init__()

        if not isinstance(module, DataParallel):
            module = DataParallel(module, device_ids)

        self.device_ids = module.device_ids
        self.dtype = parse_dtype(dtype)
        self.module = module.to(self.dtype)
        self._train_mode = True

        self.train()

    def __call__(self, *args, **kwargs):
        '''Invoke the model.
        '''
        return super().__call__(*args, **kwargs)

    def forward(self, *args, **kwargs):
        '''Applies the underlying model to a collated row of features.

        All args are cast to :class:`torch.Tensor` with the same dtype as the
        model. In evaluation mode, autograd is disabled.

        To ensure registered hooks are run, you should invoke the model object
        directly rather than calling this method.
        '''
        dtype = self.dtype
        module = self.module
        train_mode = self._train_mode

        with torch.autograd.set_grad_enabled(train_mode):
            args = list(args)
            for i, x in enumerate(args):
                if isinstance(x, np.ndarray):
                    x = torch.from_numpy(x, dtype=dtype)
                x = x.to(dtype)
                args[i] = x

            y = module(*args, **kwargs)

        return y

    def train(self, mode=True):
        '''Put the module in training mode. The opposite of :meth:`eval`.

        In training mode, autograd is enabled. The wrapped module is
        recursivly set to training mode.

        Arguments:
            mode (bool):
                If ``True``, sets the module to training mode. If ``False``,
                sets the module to evaluation mode, equivalent to :meth:`eval`.

        Returns:
            TorchModel: self
        '''
        self._train_mode = mode
        self.module.train(mode)
        return self

    def eval(self, mode=True):
        '''Put the module in evaluation mode. The opposite of :meth:`train`.

        In evaluation mode, autograd is disabled, and the wrapped module is
        recursivly set to evaluation mode.

        Arguments:
            mode (bool):
                If ``True``, sets the module to evaluation mode. If ``False``,
                sets the module to training mode, equivalent to :meth:`train`.

        Returns:
            TorchModel: self
        '''
        return self.train(not mode)
