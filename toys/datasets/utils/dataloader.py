import torch
from torch.utils.data import DataLoader as TorchDataLoader

from .utils import Zip


class DataLoader(TorchDataLoader):
    def __init__(self, *datasets, **kwargs):
        kwargs.setdefault('pin_memory', torch.cuda.is_available())
        dataset = Zip(*datasets)
        super().__init__(dataset, **kwargs)
