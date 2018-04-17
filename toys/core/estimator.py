from typing import Any, Callable, Mapping, Sequence
import logging

import abc
from abc import ABC, abstractmethod


logger = logging.getLogger(__name__)


Model = Callable


class Estimator(ABC):
    '''A useful base class for estimators.

    In toys, an estimator is any callable that returns a model. The `Estimator`
    base class provides convenient semantics for implementing estimators.

    Subclasses of `Estimator` implement a `fit` method which constructs the
    model. This method typically accepts a few required arguments (e.g. the
    training data) and many optional arguments (e.g. the hyperparameters).

    The convenience of the `Estimator` class is that the defaults for any of
    the arguments to `fit` are overridden by the arguments passed `__init__`.
    That is to say, the values for arguments are resolved in the following
    order:

    1. Arguments passed directly to the estimator (i.e. `__call__`).
    2. Keyword arguments passed to the constructor (i.e. `__init__`).
    3. Default arguments defined by `fit`.

    Note that users should not call the `fit` method but instead call the
    estimator directly.
    '''

    def __init__(self, **defaults):
        '''Construct an estimator.

        Args:
            **defaults (Mapping[str, Any]):
                Overrides the default arguments to `fit`.
        '''
        super().__init__()
        self._fit_defaults = defaults

    def __call__(self, *args, **kwargs):
        '''Construct a model, delegating to `fit`.

        Args:
            *args (Sequence):
                Passed directly to `fit`.
            **kwargs (Mapping[str, Any]):
                Passed to `fit`, possibly with different defaults.

        Returns:
            model (Model):
                The model returned by `fit`.
        '''
        for k, v in self._fit_defaults.items():
            kwargs.setdefault(k, v)
        return self.fit(*args, **kwargs)

    @abstractmethod
    def fit(self, *args, **kwargs):
        '''Constructs a model.

        The model can be any callable, and is usually some learned function.

        Subclasses must implement this method.

        Users should not call this method directly, but instead call the
        estimator itself.

        Args:
            *args (Sequence):
                Passed directly to `fit`.
            **kwargs (Mapping[str, Any]):
                Passed to `fit`, possibly with different defaults.

        Returns:
            model (Model):
                Any arbitrary callable.
        '''
        pass
