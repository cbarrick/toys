from abc import abstractmethod
from typing import Sequence

try:
    from typing import Protocol
except ImportError:
    from abc import ABC as Protocol


class Dataset(Protocol):
    '''An improved Dataset base class.
    '''
    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index):
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
        assert 0 <= max(indices) < len(dataset)
        assert 0 <= min(indices) < len(dataset)
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        i = self.indices[index]
        cols = self.dataset[i]
        return cols


class Zip(Dataset):
    '''Zip many datasets into one, like `builtins.zip`.
    '''
    def __init__(self, *datasets):
        assert 0 < len(datasets)
        for d in datasets: assert len(d) == len(datasets[0])
        self.datasets = datasets

    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, index):
        columns = []
        for dataset in self.datasets:
            x = dataset[index]
            if isinstance(x, tuple):
                columns.extend(x)
            else:
                columns.append(x)
        return tuple(columns)


class Flat(Dataset):
    '''Flatten and concatenate the columns of a dataset.

    If ``supervised=True``, then the rightmost column is flattened but not
    concatenated to the others, e.g. treat that column as the targets.
    '''
    def __init__(self, base, supervised=True):
        super().__init__()
        self.base = base
        self.supervised = supervised

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        *inputs, target = self.base[index]
        target = target.reshape(-1)
        inputs = [x.reshape(-1) for x in inputs]

        if self.supervised:
            inputs = np.concatenate(inputs)
            return inputs, target
        else:
            inputs.append(target)
            inputs = np.concatenate(inputs)
            return (inputs,)
