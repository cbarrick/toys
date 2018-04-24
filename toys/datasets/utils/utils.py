from abc import abstractmethod
from typing import Sequence


# The Protocol class introduced in Python 3.7 is an improvement over the older
# ABC class. TODO: Consider dropping support for versions older than 3.7.
# See PEP 544 <https://www.python.org/dev/peps/pep-0544/>
try:
    from typing import Protocol
except ImportError:
    from abc import ABC as Protocol


class Dataset(Protocol):
    '''An improved Dataset base class.
    '''
    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError


class Subset(Dataset):
    '''A non-empty subset of some other dataset.

    Attributes:
        dataset (Dataset):
            The source dataset.
        indices (Sequence[int]):
            The indices of elements contained in this subset.
    '''
    def __init__(self, dataset, indices):
        assert 0 < max(indices) < len(dataset)
        assert 0 < min(indices) < len(dataset)
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.dataset[self.indices[i]]


class Zip(Dataset):
    '''Zip many datasets into one, like `builtins.zip`.
    '''
    def __init__(self, *datasets):
        assert 0 < len(datasets)
        for d in datasets: assert len(d) == len(datasets[0])
        self.datasets = datasets

    def __getitem__(self, index):
        return tuple(d[index] for d in self.datasets)

    def __len__(self):
        return len(self.datasets[0])