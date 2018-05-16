from logging import getLogger
from typing import Any, Callable, Dict, Sequence, Mapping
from itertools import groupby

import numpy as np

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


def search_param_grid(grid):
    '''Iterates over parameter sets in a grid.

    Arguments:
        grid (ParamGrid or Iterable[ParamGrid] or None):
            A mapping from parameter names to sequences of allowed values, or
            an iterable of such grids. If the grid is None or empty, a single
            empty dict will be generated.
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
            yield from search_param_grid(g)


class GridSearchCV(BaseEstimator):
    def fit(self, *datasets, **kwargs):
        '''Search for the best parameters of an model.

        Arguments:
            datasets (Dataset):
                The datasets to fit.

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
            n_jobs (int or None):
                The number of worker processes. If 0, all work is done in the
                main process. If None, use the value of `os.cpu_count()`.
            dry_run (bool):
                If true, break from loops early. Useful for debugging.
            **kwargs:
                Additional keyword arguments are forwarded to the estimator
                and `DataLoader`.

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

        Todo:
            - Use a global scope for kwargs to the default score functions.
        '''
        if 'estimator' not in kwargs:
            raise ValueError('estimator not given')

        estimator = kwargs['estimator']
        param_grid = kwargs.setdefault('param_grid', None)
        cv = kwargs.setdefault('cv', 3)
        metric = kwargs.setdefault('metric', 'negative_mean_squared_error')
        n_jobs = kwargs.setdefault('n_jobs', 0)
        dry_run = kwargs.setdefault('dry_run', False)

        dataset = toys.zip(*datasets)
        metric = parse_metric(metric)

        if not callable(cv):
            cv = k_fold(cv)

        # The algorithm below follows a map-reduce pattern for parallelism:
        #
        # 1. Jobs are generated for all combinations of parameters and folds.
        # 2. Jobs executed, possibly in parallel, and mapped into scores.
        # 3. Scores are reduced into cross validation results.

        def jobs():
            for fold_number, (train, test) in enumerate(cv(dataset)):
                for param_number, params in enumerate(search_param_grid(param_grid)):
                    train_set = Subset(dataset, train)
                    test_set = Subset(dataset, test)
                    yield params, train_set, test_set, fold_number, param_number

        def run(job):
            (params, train_set, test_set, fold_number, param_number) = job
            logger.info(f'evaluating parameter set {param_number} on fold {fold_number}')
            full_params = {**kwargs, **params}
            model = estimator(train_set, **full_params)
            score = metric(model, test_set, **full_params)
            return {
                'params': tuple(params.items()),
                'score': score,
            }

        def combine(results):
            by_params = lambda x: x['params']
            by_rank = lambda x: x['mean_score']

            cv_results = []
            results = sorted(results, key=by_params)
            for params, group in groupby(results, by_params):
                scores = tuple(r['score'] for r in group)
                mean_score = np.mean(scores, axis=0)
                if not np.isscalar(mean_score):
                    mean_score = tuple(mean_score)
                result = {
                    'params': dict(params),
                    'mean_score': mean_score,
                }
                for i, score in enumerate(scores):
                    key = f'score[{i}]'
                    result[key] = score
                cv_results.append(result)

            cv_results = sorted(cv_results, key=by_rank, reverse=True)
            return cv_results

        if n_jobs == 0:
            results = (run(j) for j in jobs())
            cv_results = combine(results)
        else:
            # The 'fork' start method is required so that user scripts can
            # execute experiments at the top level. Otherwise they must be
            # protected with ``if __name__ == '__main__': ...``.
            # WARNING: Windows doesn't have process forking.
            # WARNING: Safely forking a multithreaded process is problematic.
            logger.warn('multiprocessing is not fully supported')
            ctx = mp.get_context('fork')
            with ctx.Pool(n_jobs) as pool:
                results = pool.imap(run, jobs())
                cv_results = combine(results)

        best_result = cv_results[-1]
        best_params = best_result['params']
        full_params = {**kwargs, **best_params}
        best_estimator = TunedEstimator(estimator, full_params, cv_results)

        return best_estimator
