from typing import Callable, Sequence, Mapping
from itertools import groupby

import numpy as np

import torch
from torch import multiprocessing as mp

import toys
from toys import Estimator, Model
from toys.datasets.utils import Zip, Subset
from toys.metrics import Accumulator, ScoreFn, supervised_score, unsupervised_score

from .cross_val import k_fold, CrossValSplitter


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


class GridSearchCV(Estimator):
    def fit(self, *datasets, estimator=None, param_grid=None, cv=3, metric=None,
            score_fn='supervised', n_jobs=1):
        '''Search for the best parameters of an model.

        Arguments:
            *datasets (Sequence[Dataset]):
                The datasets to which the estimator is fit.
            estimator (Estimator or None):
                The estimator which fits the model. The estimator be specified
                either when constructing or calling the `GridSearchCV` and
                MUST NOT be None. (Though None will successfully typecheck.)
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
            score_fn (str or ScoreFn):
                A function for scoring models against validation sets. The
                function will recieve a model and a subset of each dataset,
                and must return a float or tuple of floats as a score. The
                special values 'supervised' and 'unsupervised' will create
                score functions from `metric`. See `supervised_score` and
                `unsupervised_score`.
            metric (Accumulator or Sequence[Accumulator] or None):
                Accumulator(s) to measure the goodness of fit of a model. This
                argument is ignored when a custom `score_fn` is given. The
                default depends on `score_fn`.
            n_jobs (int or None):
                The number of worker processes. If 0, all work is done in the
                main process. If negative or None, the number returned by
                `os.cpu_count()` is used.

        Returns:
            model (Model):
                A model fit with the best parameters over the entire dataset.

        Raises:
            ValueError:
                The ``estimator`` argument must not be None. It must be set
                either when constructing or calling `GridSearchCV`.

        Todo:
            - Use a global scope for kwargs to the default score functions.
        '''
        assert 0 < len(datasets)

        if estimator is None:
            msg = 'Estimator must not be None.'
            msg += ' The `estimator` argument must be specified when'
            msg += ' constructing or calling `GridSearchCV` (or both).'
            raise ValueError(msg)

        if not callable(cv):
            cv = k_fold(cv)

        if score_fn == 'supervised':
            score_fn = supervised_score(metric)
        elif score_fn == 'unsupervised':
            score_fn = unsupervised_score(metric)
        else:
            assert callable(score_fn)

        if n_jobs < 0:
            n_jobs = None

        # The algorithm below follows a map-reduce pattern for parallelism:
        #
        # 1. Jobs are generated for all combinations of parameters and folds.
        # 2. Jobs executed, possibly in parallel, and mapped into scores.
        # 3. Scores are reduced into cross validation results.

        def jobs():
            for params in search_param_grid(param_grid):
                for train, test in cv(Zip(*datasets)):
                    train_sets = (Subset(d, train) for d in datasets)
                    test_sets = (Subset(d, test) for d in datasets)
                    yield params, train_sets, test_sets

        def score(job):
            (params, train_sets, test_sets) = job
            model = estimator(*train_sets, **params)
            score = score_fn(model, *test_sets)
            params = tuple(params.items())
            if not isinstance(score, tuple): score = (score,)
            return (params, score)

        def combine(scores):
            scores = sorted(scores, key=lambda x: x[0])
            scores = groupby(scores, key=lambda x: x[0])
            scores = {p: list(s) for p, s in scores.items()}
            scores = {p: tuple(np.mean(s, axis=0)) for p, s in scores.items()}
            cv_results = ({'params':p, 'mean_score':s} for p, s in scores.items())
            cv_results = sorted(cv_results, key=lambda x: x['mean_score'])
            return cv_results

        if n_jobs == 0:
            scores = (score(j) for j in jobs())
        else:
            ctx = mp.get_context('forkserver')
            pool = ctx.Pool(n_jobs)
            scores = pool.imap(score, jobs())
            pool.close()

        cv_results = combine(scores)
        best_result = cv_results[-1]
        best_params = best_result['params']
        best_model = estimator(*datasets, **best_params)
        best_model.cv_results = cv_results
        return best_model