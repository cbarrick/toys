import torch
from torch.utils.data import DataLoader as TorchDataLoader

import toys


class DataLoader(TorchDataLoader):
    def __init__(self, *datasets, **kwargs):
        valid_args = {'batch_size', 'shuffle', 'sampler', 'batch_sampler', 'num_workers',
                'collate_fn', 'pin_memory', 'drop_last', 'timeout', 'worker_init_fn'}

        kwargs.setdefault('pin_memory', torch.cuda.is_available())
        kwargs = {k: kwargs[k] for k in valid_args if k in kwargs}

        dataset = toys.zip(*datasets)

        super().__init__(dataset, **kwargs)
