from typing import Any, Callable, Mapping, Sequence
import logging

import abc
from abc import ABC, abstractmethod

import toys
from toys.datasets.utils import Dataset


logger = logging.getLogger(__name__)


Model = Callable


class Estimator(ABC):
    '''A useful base class for estimators.

    An estimator is any callable that returns a model. The `Estimator` base
    class provides a convenient API for implementing estimators.

    Subclasses of `Estimator` implement a `fit` method which constructs the
    model. This method typically accepts a few required arguments (e.g. the
    training data) and many optional arguments (e.g. the hyperparameters).

    Instances of `Estimator` have a dictionary parameter `defaults`. When the
    estimator is called as a function, it delegates to the `fit` method using
    this mapping as default keyword arguments. This means that you likely
    never want to call the `fit` method diretly.
    '''

    def __init__(self, **defaults):
        '''Construct an estimator.

        Arguments:
            **defaults (Mapping[str, Any]):
                Overrides the default hyperparameters.
        '''
        super().__init__()
        self.defaults = defaults

    def __call__(self, *datasets, **params):
        '''Construct a model, delegating to `fit`.

        Arguments:
            *datasets (Dataset):
                Passed directly to `fit`.
            **params:
                Passed to `fit`, with defaults taken from ``self.defaults``.

        Returns:
            model (Model):
                The model returned by `fit`.
        '''
        params = self.defaults.copy().update(params)
        return self.fit(*datasets, **params)

    @abstractmethod
    def fit(self, *datasets, **params):
        '''Constructs a model.

        Subclasses must implement this method.

        The return value can be any callable, and is usually some learned
        function. Meta-estimators like `GridSearchCV` return other estimators.

        When the estimator is called as a function, it delegates to the `fit`
        method using ``self.defaults`` as default keyword arguments. This
        means that you likely never want to call `fit` diretly.

        Arguments:
            *datasets (Dataset):
                The datasets required to train the model.
            **params:
                The hyperparameters to use while training the model.

        Returns:
            model (Model):
                Any arbitrary callable.
        '''
        pass
