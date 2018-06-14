from logging import getLogger
from typing import Callable, Iterable, Mapping, Sequence, Tuple

import numpy as np

import toys
from toys.common import BaseEstimator
from toys.data import Dataset


logger = getLogger(__name__)


Fold = Tuple[Dataset, Dataset]
CrossValSplitter = Callable[[Dataset], Iterable[Fold]]
ParamGrid = Mapping[str, Sequence]


def combinations(grid):
    '''Iterates over all combinations of parameters in a parameter grid.

    A parameter grid is a mapping from parameter names to a sequence of
    possible values. This function yields dictionaries mapping all names in the
    grid to exactly one value, for all possible combinations. The argument
    ``grid`` may also be a sequence of parameter grids, which is equivalent to
    chaining the iterators for each individual grid. If a ``grid`` is
    :obj:`None`, this function yields a single empty dictionary.

    Arguments:
        grid (ParamGrid or Iterable[ParamGrid] or None):
            One or more parameter grids to search.

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


class KFold(CrossValSplitter):
    '''A splitter for simple k-fold cross validation.

    K-folding partitions a dataset into k subsets of roughly equal size. Each
    "fold" is a pair of datasets, ``(train, test)``, where ``test`` is one of
    the partitions and ``train`` is the concatenation of the remaining
    partitions.

    Instances of this class are functions which apply k-folding to datasets.
    They return an iterator over all folds of the datasets.

    If ``shuffle`` is true, the elements of each partition are chosen at
    random. Otherwise each partition is a continuous subset of the dataset.

    Arguments:
        k (int):
            The number of folds. Must be at least 2.
        shuffle (bool):
            Whether to shuffle the indices before splitting.
    '''

    def __init__(self, k=3, shuffle=True):
        if k < 2:
            raise ValueError('The number of folds must be at least 2.')

        self.k = k
        self.shuffle = shuffle

    def __call__(self, dataset):
        indices = np.arange(len(dataset))
        if self.shuffle: np.random.shuffle(indices)
        splits = np.array_split(indices, self.k)
        for test_indices in splits:
            train_indices = [s for s in splits if s is not test_indices]
            train_indices = np.concatenate(train_indices)
            train = toys.subset(dataset, train_indices)
            test = toys.subset(dataset, test_indices)
            yield train, test


class TunedEstimator(BaseEstimator):
    '''An estimator wrapped with a with default kwargs.

    These are often returned by meta-estimators performain a parameter search,
    e.g. :class:`~toys.model_selection.GridSearchCV`.

    Attributes:
        estimator (Estimator):
            The underlying estimator.
        kwargs (Dict[str, Any]):
            Overrides for the default kwargs of the estimator.
        cv_results (pandas.DataFrame or Dict or None):
            An optional table attached to the instance.
    '''
    def __init__(self, estimator, kwargs, cv_results=None):
        super().__init__()
        self.estimator = estimator
        self.kwargs = kwargs
        self.cv_results = pd.DataFrame(cv_results)

    def fit(self, *args, **kwargs):
        kwargs = {**self.kwargs, **kwargs}
        model = self.estimator(*args, **kwargs)
        return model
