from abc import abstractmethod
from typing import *


# The Protocol type does not exist until Python 3.7.
# TODO: Remove the try-except when Python 3.6 support is dropped.
try:
    from typing import Protocol
except ImportError:
    from abc import ABC as Protocol


class Dataset(Protocol):
    '''The Dataset protocol.
    '''
    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError


ArrayShape = Tuple[int, ...]
DatasetShape = Union[Tuple[Union[ArrayShape, None], ...], None]

CrossValSplitter = Callable[[Dataset], Iterable[Tuple[Sequence[int], Sequence[int]]]]
Estimator = Callable
Metric = Callable
Model = Callable
ParamGrid = Mapping[str, Sequence]
