import torch
from torch.utils.data import DataLoader as TorchDataLoader

from .utils import Zip


class DataLoader(TorchDataLoader):
    def __init__(self, *datasets, **kwargs):
        kwargs.setdefault('pin_memory', torch.cuda.is_available())
        if 1 < len(datasets):
            dataset = Zip(*datasets)
        else:
            dataset = datasets[0]
        super().__init__(dataset, **kwargs)
