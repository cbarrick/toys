import numpy as np

import toys
from toys.typing import CrossValSplitter, Dataset


class KFold(CrossValSplitter):
    '''A splitter for simple k-fold cross validation.

    K-folding partitions a dataset into k subsets of roughly equal size. A
    "fold" is a pair of datasets, ``(train, test)``, where ``test`` is one of
    the partitions and ``train`` is the concatenation of the remaining
    partitions.

    Instances of this class are functions which apply k-folding to datasets.
    They return an iterator over all folds of the datasets.
    '''

    def __init__(self, k=3, shuffle=True):
        '''Initialize a KFold.

        If ``shuffle`` is true, the elements of each partition are chosen at
        random. Otherwise each partition is a continuous subset of the dataset.

        Arguments:
            k (int):
                The number of folds. Must be at least 2.
            shuffle (bool):
                Whether to shuffle the indices before splitting.
        '''
        if k < 2:
            raise ValueError('The number of folds must be at least 2.')

        self.k = k
        self.shuffle = shuffle

    def __call__(self, dataset):
        indices = np.arange(len(dataset))
        if self.shuffle: np.random.shuffle(indices)
        splits = np.array_split(indices, self.k)
        for test in splits:
            train = [s for s in splits if s is not test]
            train = np.concatenate(train)
            yield toys.subset(dataset, train), toys.subset(dataset, test)
