from logging import getLogger
from typing import Sequence

import pandas as pd

import toys
from toys.common import BaseEstimator, Estimator
from toys.metrics import Metric
from toys.parsers import parse_metric

from .core import CrossValSplitter, KFold, ParamGrid, TunedEstimator


logger = getLogger(__name__)


class GridSearchCV(BaseEstimator):
    '''A meta-estimator to exhaustively search a grid of hyperparameters.
    '''

    def fit(self, *datasets, estimator=None, param_grid=None, cv=3, metric='f_score',
            minimize=False, n_jobs=0, dry_run=False):
        '''Learn the best hyper-parameters of an estimator.

        Arguments:
            datasets (Dataset):
                The datasets to fit. If more than one are given, they are
                combined using :func:`toys.zip`.

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
            metric (str or Metric or Sequence[str or Metric]):
                A metric or metrics to measure the goodness of fit of a model.
            minimize (bool):
                Set to true to choose the parameters which score the lowest.
            n_jobs (int or None):
                The number of worker processes. If 0, all work is done in the
                main process. If None, use the value of ``os.cpu_count()``.
            dry_run (bool):
                If true, break from loops early. Useful for debugging.

        Returns:
            TunedEstimator:
                An estimator which defaults to using the best hyperparameters
                found through cross validated grid search. The estimator has
                an attribute :attr:`cv_results` which contains the outcomes
                of the grid search.

        Raises:
            ValueError:
                The ``estimator`` argument must not be :obj:`None`. It
                must be set either when constructing or invoking a
                :class:`GridSearchCV`.
        '''
        dataset = toys.zip(*datasets)
        metric = parse_metric(metric)

        if estimator is None:
            raise TypeError('estimator must not be None')

        if n_jobs != 0:
            logger.warn('multiprocessing is not yet supported')

        if not callable(cv):
            cv = KFold(cv)

        def jobs():
            for train, test in cv(dataset):
                for params in combinations(param_grid):
                    yield params, train, test

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
