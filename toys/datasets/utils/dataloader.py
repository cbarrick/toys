import torch
from torch.utils.data import DataLoader as TorchDataLoader

import toys
from toys.datasets.utils import Zip


class DataLoader(TorchDataLoader):
    def __init__(self, *datasets, **kwargs):
        device_count = torch.cuda.device_count()
        device_ids = list(range(device_count))
        kwargs.setdefault('pin_memory', len(device_ids) > 0)
        dataset = Zip(*datasets)
        super().__init__(dataset, **kwargs)
