import logging
from pathlib import Path

import torch
import torch.utils.data

from torchvision.datasets import MNIST as TorchMNIST
from torchvision.transforms import ToTensor


logger = logging.getLogger(__name__)


class MNIST:
    def __init__(self, path='./data/mnist'):
        self.path = path

    def load(self):
        train = TorchMNIST(self.path, train=True, download=True, transform=ToTensor())
        test = TorchMNIST(self.path, train=False, download=True, transform=ToTensor())
        return train, test
