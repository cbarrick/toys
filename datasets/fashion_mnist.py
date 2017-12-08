import logging
from pathlib import Path

import torch
import torch.utils.data

from torchvision.datasets import FashionMNIST as TorchFashionMNIST
from torchvision.transforms import ToTensor


logger = logging.getLogger(__name__)


class FashionMNIST:
    def __init__(self, path='./data/mnist'):
        self.path = path

    def load(self):
        train = TorchFashionMNIST(self.path, train=True, download=True, transform=ToTensor())
        test = TorchFashionMNIST(self.path, train=False, download=True, transform=ToTensor())
        return train, test
