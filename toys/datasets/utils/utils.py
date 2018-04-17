from abc import ABC, abstractmethod

import torch
from torch.utils.data import Dataset as TorchDataset


class Dataset(ABC, TorchDataset):
    '''A wrapper around `torch.utils.data.Dataset`
    with proper abstract method declarations.
    '''
    def __init__(self):
        # ABC and TorchDataset are not cooperative with
        # multiple inheritance, so we initialize them manually.
        ABC.__init__(self)
        Dataset.__init__(self)

    @abstractmethod
    def __getitem__(self, index):
        pass

    @abstractmethod
    def __len__(self):
        pass


class Zip(Dataset):
    '''Zip many datasets into one, like `builtins.zip`.
    '''
    def __init__(self, *datasets):
        assert len(datasets) > 0
        for d in datasets: assert len(d) == len(datasets[0])
        self.datasets = datasets

    def __getitem__(self, index):
        return tuple(d[index] for d in self.datasets)

    def __len__(self):
        return len(self.datasets[0])
