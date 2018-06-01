from logging import getLogger
from typing import Any, Callable, Dict, Sequence, Mapping
from itertools import groupby

import numpy as np
import pandas as pd

import torch
from torch import multiprocessing as mp

import toys
from toys.common import BaseEstimator, Estimator, Model, TunedEstimator
from toys.datasets.utils import DataLoader, Subset
from toys.metrics import Metric
from toys.parsers import parse_metric

from .cross_val import k_fold, CrossValSplitter


logger = getLogger(__name__)


ParamGrid = Mapping[str, Sequence]


def combinations(grid):
    '''Iterates over all combinations of parameters in the grid.

    Arguments:
        grid (ParamGrid or Iterable[ParamGrid] or None):
            A parameter grid, list of parameter grids or None. A parameter grid
            is a mapping from parameter names to sequences of allowed values.

    Yields:
        A dictionary mapping parameter names to values.
    '''
    if not grid:
        yield {}

    elif isinstance(grid, Mapping):
        indices = {k:0 for k in grid.keys()}
        lens = {k:len(v) for k, v in grid.items()}
        n = int(np.prod(list(lens.values())))
        for _ in range(n):
            yield {k: grid[k][indices[k]] for k in grid}
            for k in indices:
                indices[k] += 1
                if indices[k] == lens[k]:
                    indices[k] = 0
                else:
                    break

    else:
        for g in grid:
            yield from combinations(g)


class GridSearchCV(BaseEstimator):
    def fit(self, *datasets, estimator=None, param_grid=None, cv=3, metric='f_score',
            minimize=False, n_jobs=0, dry_run=False):
        '''Learn the best hyper-parameters of an estimator.

        Arguments:
            datasets (Dataset):
                The datasets to fit. If more than one are given, they are
                combined using `toys.zip`.

        Keyword Arguments:
            estimator (Estimator or None):
                The inner estimator which fits the model. The inner estimator
                may be specified either when constructing or calling this
                estimator and MUST NOT be None. (Though None will successfully
                typecheck.)
            param_grid (ParamGrid or Iterable[ParamGrid] or None):
                A mapping from parameter names to sequence of allowed values,
                or an iterable of such grids. A value of None will fit a
                single model using the default parameters.
            cv (int or CrossValSplitter):
                The cross validation strategy to use. If an int is given, a
                shuffled k-fold cross validation is used with this many folds.
                Otherwise, this should be a function which accepts a dataset
                and returns an iterable over ``(train, test)`` pairs, where
                ``train`` indexes the training instances and ``test`` indexes
                the validation instances.
            metric (str or Metric or Sequence[str or Metric]):
                A metric or metrics to measure the goodness of fit of a model.
            minimize (bool):
                Set to true to choose the parameters which score the lowest.
            n_jobs (int or None):
                The number of worker processes. If 0, all work is done in the
                main process. If None, use the value of `os.cpu_count()`.
            dry_run (bool):
                If true, break from loops early. Useful for debugging.

        Returns:
            best_estimator (TunedEstimator):
                An estimator which defaults to using the best hyperparameters
                found through cross validated grid search. The estimator has
                an attribute `cv_results` which contains the overall results
                of the grid search.

        Raises:
            ValueError:
                The ``estimator`` argument must not be None. It must be set
                either when constructing or calling `GridSearchCV`.
        '''
        dataset = toys.zip(*datasets)
        metric = parse_metric(metric)

        if estimator is None:
            raise TypeError('estimator must not be None')

        if n_jobs != 0:
            logger.warn('multiprocessing is not yet supported')

        if not callable(cv):
            cv = k_fold(cv)

        def jobs():
            for train, test in cv(dataset):
                for params in combinations(param_grid):
                    train_set = Subset(dataset, train)
                    test_set = Subset(dataset, test)
                    yield params, train_set, test_set

        def run(job):
            (params, train_set, test_set) = job
            model = estimator(train_set, **params)
            score = float(metric(model, test_set))
            params = tuple(sorted(params))
            return {'score':score, 'params':params}

        def combine(results):
            results = pd.DataFrame(results)
            results = results.groupby('params').mean().reset_index()
            results = results.sort_values('score', ascending=minimize)
            results['params'] = results['params'].apply(dict)
            return results

        results = (run(j) for j in jobs())
        cv_results = combine(results)

        best_result = cv_results.iloc[0]
        best_params = best_result['params']
        best_estimator = TunedEstimator(estimator, best_params, cv_results)

        return best_estimator
