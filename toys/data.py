from abc import abstractmethod
from io import StringIO
from typing import *

import numpy as np

import torch
from torch.utils.data import DataLoader


# The Protocol type does not exist until Python 3.7.
# TODO: Remove the try-except when Python 3.6 support is dropped.
try:
    from typing import Protocol
except ImportError:
    from abc import ABC as Protocol


ColumnShape = Optional[Tuple[Optional[int], ...]]
RowShape = Optional[Tuple[ColumnShape, ...]]


class Dataset(Protocol):
    '''The dataset protocol.
    '''
    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError


def common_shape(shape1, shape2):
    '''Compute the most immediate common shape of the inputs.

    E.g.::

    >>> common_shape(shape1=(1, (1, 1)), shape2=(2, (1, 1)))
    (None, (1, 1))
    >>> common_shape(shape1=(1, (1, 1)), shape2=(1, (2, 1)))
    (1, (None, 1))
    >>> common_shape(shape1=(1, (1, 1)), shape2=(1, (1, 1, 2)))
    (1, None)
    >>> common_shape(shape1=(1, (1, 1)), shape2=(1, (1, 1), 2))
    None

    Arguments:
        shape1 (VariableShape):
            The first shape.
        shape2 (VariableShape):
            The second shape.

    Returns:
        VariableShape:
            The least general (i.e. most specialized) shape for which both
            input shapes are instances.
    '''
    if shape1 == shape2:
        return shape1
    if np.isscalar(shape1) or np.isscalar(shape2):
        return None
    if len(shape1) != len(shape2):
        return None
    return tuple(common_shape(a, b) for a, b in zip(shape1, shape2))


def shape(dataset):
    '''Infer the shape of the dataset.

    This function will sample up to four rows from the dataset to identify
    if any part of the shape is variable.

    Arguments:
        dataset (Dataset):
            The dataset whose shape will be checked.

    Returns:
        RowShape:
            A tuple of shapes, one for each column. If any part of the shape
            is variable, it is replaced by :obj:`None`.

    Example:
        >>> from toys.datasets import SimulatedLinear
        >>> a = SimulatedLinear(100, in_shape=(32,32,3), out_shape=10)
        >>> toys.shape(a)
        ((32, 32, 3), (10,))

        .. todo::
            The example does not run.
    '''
    n = len(dataset)
    if n == 0: return None

    row1 = dataset[np.random.randint(n)]
    row2 = dataset[np.random.randint(n)]
    row3 = dataset[np.random.randint(n)]
    row4 = dataset[np.random.randint(n)]

    shape1 = tuple(np.shape(col) for col in row1)
    shape2 = tuple(np.shape(col) for col in row2)
    shape3 = tuple(np.shape(col) for col in row3)
    shape4 = tuple(np.shape(col) for col in row4)

    shape5 = common_shape(shape1, shape2)
    shape6 = common_shape(shape3, shape4)

    return common_shape(shape5, shape6)


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

    def __repr__(self):
        return f'Subset({repr(self.dataset)}, {repr(self.indices)})'

    @property
    def hints(self):
        return getattr(self.dataset, 'hints', {})


def subset(dataset, indices):
    '''Select a subset of some dataset by row indices.

    Arguments:
        dataset (Dataset):
            The source dataset.
        indices (Sequence[int]):
            The indices of elements contained in this subset.

    Returns:
        Dataset:
            A subset of the input.

    Example:
        >>> from toys.datasets import SimulatedLinear
        >>> a = SimulatedLinear(100)
        >>> len(a)
        100
        >>> b = toys.subset(a, np.arange(0, 50))
        >>> len(b)
        50
    '''
    return Subset(dataset, indices)


class Zip(Dataset):
    '''Combines the columns of many datasets into one.
    '''
    def __init__(self, *datasets):
        if len(datasets) == 0:
            raise TypeError('Zip() requires at least 1 dataset.')

        for d in datasets:
            if len(d) != len(datasets[0]):
                raise ValueError('Zip() requires all datasets to be the same length.')

        self.datasets = datasets

    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, index):
        columns = []
        for dataset in self.datasets:
            x = dataset[index]
            columns.extend(x)
        return tuple(columns)

    def __repr__(self):
        buf = StringIO()
        buf.write('Zip(')
        datasets = (repr(ds) for ds in self.datasets)
        print(*datasets, sep=', ', end=')', file=buf)
        return buf.getvalue()

    @property
    def hints(self):
        ret = {}
        for ds in reversed(self.datasets):
            sub = getattr(ds, 'hints', {})
            ret.update(sub)
        return ret


# This is reexported as toys.zip.
# The underscore is used here to prevent overriding builtins.zip.
def zip_(*datasets):
    '''Returns a dataset with all of the columns of the given datasets.

    Arguments:
        datasets (Dataset):
            The datasets to combine.

    Returns:
        Dataset:
            The combined dataset.

    Example:
        >>> from toys.datasets import SimulatedLinear
        >>> a = SimulatedLinear(100, in_shape=4, out_shape=5)
        >>> b = SimulatedLinear(100, in_shape=6, out_shape=7)
        >>> c = toys.zip(a, b)
        >>> len(a) == len(b)
        True
        >>> toys.shape(a)
        ((4,), (5,))
        >>> toys.shape(b)
        ((6,), (7,))
        >>> toys.shape(c)
        ((4,), (5,), (6,), (7,))
    '''
    if len(datasets) == 0:
        raise TypeError('zip() requires at least 1 dataset.')
    if len(datasets) == 1:
        return datasets[0]
    else:
        return Zip(*datasets)


class Concat(Dataset):
    '''Combines the rows of many datasets into one.
    '''
    def __init__(self, *datasets):
        if len(datasets) == 0:
            raise TypeError('Concat() requires at least 1 dataset.')

        self.lens = tuple(len(d) for d in datasets)
        self.datasets = datasets

    def __len__(self):
        return sum(self.lens)

    def __getitem__(self, index):
        for i, n in enumerate(self.lens):
            if n <= index:
                index -= n
            else:
                return self.datasets[i][index]

    def __repr__(self):
        buf = StringIO()
        buf.write('Concat(')
        datasets = (repr(ds) for ds in self.datasets)
        print(*datasets, sep=', ', end=')', file=buf)
        return buf.getvalue()

    @property
    def hints(self):
        ret = {}
        for ds in reversed(self.datasets):
            sub = getattr(ds, 'hints', {})
            ret.update(sub)
        return ret


def concat(*datasets):
    '''Returns a dataset with all of the rows of the given datasets.

    Arguments:
        datasets (Dataset):
            The datasets to combine.

    Returns:
        Dataset:
            The combined dataset.

    Example:
        >>> from toys.datasets import SimulatedLinear
        >>> a = SimulatedLinear(100)
        >>> b = SimulatedLinear(200)
        >>> c = toys.concat(a, b)
        >>> toys.shape(a) == toys.shape(b) == toys.shape(c)
        True
        >>> len(a)
        100
        >>> len(b)
        200
        >>> len(c)
        300
    '''
    if len(datasets) == 0:
        raise TypeError('concat() requires at least 1 dataset.')
    if len(datasets) == 1:
        return datasets[0]
    else:
        return Concat(*datasets)


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
        row = self.base[index]
        row = [x.reshape(-1) for x in row]

        if self.supervised:
            *features, target = row
            features = np.concatenate(features)
            return features, target
        else:
            features = np.concatenate(row)
            return (features,)

    def __repr__(self):
        return f'Flat({repr(self.base)}, supervised={repr(self.supervised)})'

    @property
    def hints(self):
        return self.base.hints


def flatten(dataset, supervised=True):
    '''Returns a dataset whose columns are flattened and concatenated together.

    In supervised mode, the rightmost column is flattened but is kept as a
    separate column. This is for supervised estimators which expect a target
    value in a separate column.

    Arguments:
        dataset (Dataset):
            The dataset to flatten.
        supervised (bool):
            Operate in supervised mode.

    Returns:
        Dataset:
            The combined dataset. If supervised is False, the dataset contains
            a single column with a flat shape. If supervised is True, the
            dataset contains two columns with flat shape.

    Example:
        >>> a = SimulatedLinear(100, in_shape=(32,32,3), out_shape=(32,32,15))
        >>> toys.shape(a)
        ((32, 32, 3), (32, 32, 15))
        >>> b = toys.flatten(a)
        >>> toys.shape(b)
        ((3072,), (15360,))

        .. todo::
            The example does not run.
    '''
    cols = dataset[0]

    if supervised:
        assert 2 <= len(cols)

    if 3 <= len(cols):
        return Flat(dataset, supervised)

    if 2 == len(cols) and not supervised:
        return Flat(dataset, supervised)

    for col in cols:
        if len(col.shape) != 1:
            return Flat(dataset, supervised)

    # If we've got this far, the dataset is already flat
    return dataset


def batches(dataset, batch_size=None, **kwargs):
    '''Iterates over a dataset in batches.

    If the dataset has an attribute :attr:`~Dataset.hints`, then it must be a
    dictionary mapping argument names to recommended values.

    .. seealso::
        See the :doc:`/guides/datasets` user guide for information on batching
        and argument hinting.

    Arguments:
        dataset (Dataset):
            The dataset to iterate over.
        batch_size (int):
            The maximum size of the batches.

    Keyword Arguments:
        **kwargs:
            Keyword arguments are forwarded to |DataLoader|.

    Returns:
        torch.utils.data.DataLoader:
            An iteratable over batches of the dataset.

    Example:
        .. todo::
            Add an example.

    .. |DataLoader| replace::
        :class:`~torch.utils.data.DataLoader`
    '''
    if batch_size is not None:
        kwargs.setdefault('batch_size', batch_size)

    hints = getattr(dataset, 'hints', {})
    kwargs = {**hints, **kwargs}

    kwargs.setdefault('pin_memory', torch.cuda.is_available())
    kwargs.setdefault('batch_size', batch_size)
    return DataLoader(dataset, **kwargs)
