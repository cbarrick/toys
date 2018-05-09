from abc import ABC, abstractmethod
from logging import getLogger
from pathlib import Path

import numpy as np
import torch

import toys
from toys.datasets.utils import Dataset


logger = getLogger(__name__)


class _CIFAR(Dataset, ABC):
    '''A base class for the CIFAR datasets.

    Subclasses must implement the `_read_from_disk` method which fetches the
    image and label from disk. Subclasses should also provide class attributes
    ``url`` and ``md5`` corresponding to an archive containing the data.
    '''

    def __init__(self, base_path='./data', test=False, prefetch=True):
        '''Construct a CIFAR dataset.

        Arguments:
            base_path (str or Path):
                A directory in which to store the data.
            test (bool):
                Set true to load the test set.
            prefetch (bool):
                Set false to stream the dataset from disk.
        '''
        super().__init__()
        self.base_path = Path(base_path)
        self.test = test

        name = f"{self.__class__.__name__} {'test' if test else 'train'} set"

        if prefetch:
            logger.info(f'prefetching {name}')
            data, labels = self._read_all_from_disk()
            self.data = data
            self.labels = labels
            logger.info(f'prefetching {name} complete')

        else:
            logger.info(f'streaming {name}')
            self.data = None
            self.labels = None

    def __len__(self):
        '''The number of images in the dataset.

        Returns:
            length (int):
                Either 50,000 for the train set or 10,000 for the test set
        '''
        if self.test:
            return 10000
        else:
            return 50000

    def __getitem__(self, index):
        '''Load a single instance by index.

        Returns:
            img (np.ndarray):
                A float32 color image of shape (32, 32, 3).
            label (int):
                A class label.
        '''
        if self.data is None:
            img, label = self._read_from_disk(index)
            return img, label
        else:
            img = self.data[index]
            label = self.labels[index]
            return img, label

    def _read_all_from_disk(self):
        '''Read all data from disk into congiguous arrays.

        Returns:
            data (np.ndarray):
                A float32 array of all images, of shape (n, 32, 32, 3).
            labels (np.ndarray):
                An int64 array of all labels, of shape (n,).
        '''
        n = len(self)
        data = np.ndarray((n, 32, 32, 3), dtype='float32')
        labels = np.ndarray(n, dtype='int64')

        for i in range(n):
            img, label = self._read_from_disk(i)
            data[i] = img
            labels[i] = label

        return data, labels

    def _download_and_verify(self):
        # TODO: This should download, verify, and extract the dataset.
        # It should not download if the archive exists and is verified.
        # Use `self.url` and `self.md5`.
        # This should be run during `__init__` before `_read_all_from_disk`.
        raise NotImplementedError

    @property
    @abstractmethod
    def url(self):
        '''The URL of the dataset.'''
        raise NotImplementedError

    @property
    @abstractmethod
    def md5(self):
        '''The md5sum of the dataset.'''
        raise NotImplementedError

    @abstractmethod
    def _read_from_disk(self, index):
        '''Read a single instance from disk.

        Arguments:
            index (int):
                The index of an image.

        Returns:
            img (np.ndarray):
                A float32 color image of shape (32, 32, 3).
            label (int):
                A class label.
        '''
        raise NotImplementedError


class CIFAR10(_CIFAR):
    '''The CIFAR-10 dataset.

    The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes,
    with 6000 images per class. There are 50000 training images and 10000 test
    images. The training images contain exactly 5000 images from each class.

    See https://www.cs.toronto.edu/~kriz/cifar.html.
    '''

    url = 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
    md5 = 'c32a1d4ab5d03f1284b67883e8d87530'

    def _read_from_disk(self, index):
        batch = index // 10000
        index = index - (10000 * batch)

        prefix = 'cifar-10-batches-bin'
        filename = 'test_batch.bin' if self.test else f'data_batch_{batch+1}.bin'
        path = self.base_path / prefix / filename

        with open(path, 'rb') as f:
            f.seek(index * 3073)
            b = f.read(3073)
            label = b[0]
            img = b[1:]
            img = np.frombuffer(img, 'uint8').reshape(3, 32, 32)
            img = np.einsum('chw->hwc', img)
            img = img.astype('float32') / 255
            return img, label


class CIFAR20(_CIFAR):
    '''The CIFAR-20 dataset.

    The CIFAR-20 dataset consists of 60000 32x32 colour images in 20 classes.
    There are 50000 training images and 10000 test images. The images are the
    same as the CIFAR-100 dataset, but labeled into 20 coarser classes.

    See https://www.cs.toronto.edu/~kriz/cifar.html.
    '''

    url = 'https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz'
    md5 = '03b5dce01913d631647c71ecec9e9cb8'

    def _read_from_disk(self, index):
        prefix = 'cifar-100-binary'
        filename = 'test.bin' if self.test else 'train.bin'
        path = self.base_path / prefix / filename

        with open(path, 'rb') as f:
            f.seek(index * 3074)
            b = f.read(3074)
            label = b[0]
            img = b[2:]
            img = np.frombuffer(img, 'uint8').reshape(3, 32, 32)
            img = np.einsum('chw->hwc', img)
            img = img.astype('float32') / 255
            return img, label


class CIFAR100(_CIFAR):
    '''The CIFAR-100 dataset.

    The CIFAR-100 dataset consists of 60000 32x32 colour images in 100 classes,
    with 600 images per class. There are 50000 training images and 10000 test
    images. There are 500 training images and 100 testing images per class.

    See https://www.cs.toronto.edu/~kriz/cifar.html.
    '''

    url = 'https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz'
    md5 = '03b5dce01913d631647c71ecec9e9cb8'

    def _read_from_disk(self, index):
        prefix = 'cifar-100-binary'
        filename = 'test.bin' if self.test else 'train.bin'
        path = self.base_path / prefix / filename

        with open(path, 'rb') as f:
            f.seek(index * 3074)
            b = f.read(3074)
            label = b[1]
            img = b[2:]
            img = np.frombuffer(img, 'uint8').reshape(3, 32, 32)
            img = np.einsum('chw->hwc', img)
            img = img.astype('float32') / 255
            return img, label
