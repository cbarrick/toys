from logging import getLogger
from typing import Any, Callable, Dict, Sequence, Mapping
from itertools import groupby

import numpy as np

import torch
from torch import multiprocessing as mp

import toys
from toys.common import BaseEstimator, Estimator, Model, TunedEstimator
from toys.datasets.utils import DataLoader, Subset
from toys.metrics import Accumulator, NegMeanSquaredError
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
    def fit(self, dataset, *, estimator=None, param_grid=None, cv=3, n_jobs=0, dry_run=False,
            metric='negative_mean_squared_error', **kwargs):
        '''Search for the best parameters of an model.

        Arguments:
            dataset (Sequence[Dataset]):
                The dataset to fit.

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
            metric (Accumulator or Sequence[Accumulator]):
                A metric or metrics to measure the goodness of fit of a model.
            n_jobs (int or None):
                The number of worker processes. If 0, all work is done in the
                main process. If None, use the value of `os.cpu_count()`.
            dry_run (bool):
                If true, break from loops early. Useful for debugging.
            **kwargs:
                Forwarded to the `DataLoader`.

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
        if estimator is None:
            msg = 'Estimator must not be None.'
            msg += ' The `estimator` argument must be specified when'
            msg += ' constructing or calling `GridSearchCV` (or both).'
            raise ValueError(msg)

        if not callable(cv):
            cv = k_fold(cv)

        if isinstance(metric, (Accumulator, str)):
            metric = [metric]
        metric = [parse_metric(m) for m in metric]

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
            model = estimator(train_set, **params)

            for *inputs, target in DataLoader(test_set, **kwargs):
                prediction = model(*inputs)
                for m in metric:
                    m.accumulate(target, prediction)
                if dry_run: break
            score = tuple(m.reduce() for m in metric)

            params = tuple(params.items())
            return {'params':params, 'score':score}

        def combine(results):
            by_params = lambda x: x['params']
            by_rank = lambda x: x['mean_score']

            cv_results = []
            results = sorted(results, key=by_params)
            for params, group in groupby(results, by_params):
                scores = [result['score'] for result in group]
                mean_score = np.mean(scores, axis=0)
                cv_results.append({
                    'params': dict(params),
                    'mean_score': tuple(mean_score),
                    'scores': scores,
                })

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
        best_estimator = TunedEstimator(estimator, best_params, cv_results)

        return best_estimator
