from typing import Callable, Iterable, Sequence, Tuple

import numpy as np

import toys
from toys.common import Dataset


CrossValSplitter = Callable[[Dataset], Iterable[Tuple[Sequence[int], Sequence[int]]]]


def k_fold(k=3, shuffle=True):
    '''Returns a splitter function for k-fold cross validation.

    K-folding partitions a dataset into k subsets of roughly equal size.

    If ``shuffle`` is true, the elements of each partition are chosen at
    random. Otherwise each partition is a continuous subset of the dataset.

    Arguments:
        k (int):
            The number of folds. Must be at least 2.
        shuffle (bool):
            Whether to shuffle the indices before splitting.

    Returns:
        cv (CrossValSplitter):
            A function which takes a dataset and returns an iterator over pairs
            of lists of indices, ``(train, test)``, where ``train`` indexes the
            training instances of the fold and ``test`` indexes the testing
            instances.
    '''
    assert 1 < k, 'The number of folds must be at least 2.'

    def cv(dataset):
        indices = np.arange(len(dataset))
        if shuffle: np.random.shuffle(indices)
        splits = np.array_split(indices, k)
        for test in splits:
            train = [s for s in splits if s is not test]
            train = np.concatenate(train)
            yield train, test

    return cv
