import logging
import re
import tarfile
from pathlib import Path

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import requests

import torch
import torch.utils.data


logger = logging.getLogger(__name__)


def get_data(path='./data/epi', force_download=False):
    '''Download the data and return a Path to the directory.
    '''
    src = 'http://andrewjanowczyk.com/wp-static/epi.tgz'
    arc = Path('./epi.tgz')
    dst = Path(path)

    if dst.exists() and not force_download:
        return dst

    logger.info(f'downloading {src}')
    r = requests.get(src, stream=True)
    with arc.open('wb') as fd:
        for chunk in r.iter_content(chunk_size=128):
            fd.write(chunk)
    dst.mkdir(parents=True, exist_ok=True)
    tarfile.open(arc).extractall(dst)
    arc.unlink()
    return dst


def get_metadata(image_path):
    '''Get the metadata for an image given its path.
    '''
    image_path = str(image_path)
    match = re.search('([0-9]+)_([0-9]+)', image_path)
    return {
        'id': match[2],
        'patient': match[1],
        'path': image_path,
    }


def imread(path, mask=False):
    '''Open an image as a numpy array.

    Masks are opened as binary images with shape (N x M).
    Otherwise, images are opened as float RGB with shape (N x M x 3).
    '''
    path = str(path)
    if mask:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        image = image.astype('bool')
    else:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        image = image[:, :, [2,1,0]] # BGR to RGB
        image = image.astype('float32') / 255
    return image


def background_mask(mask_p):
    '''Create a mask separating the background from the foreground.
    '''
    mask_b = np.logical_not(mask_p)
    return mask_b


def edge_mask(mask_p, size=5):
    '''Creates a mask around the outer edges of the positive class.
    '''
    k = np.ones((size, size))
    mask_p = mask_p.astype('uint8')
    edges = cv2.dilate(mask_p, k)
    edges = edges - mask_p
    mask_e = edges.astype('bool')
    return mask_e


def extract_from_mask(image, mask, max_count=None, size=64, random=True):
    '''Sample patches from an image whose centers are not masked.
    '''
    ar = np.require(image) # no copy
    mask = np.array(mask)  # copy
    width, height = mask.shape
    delta = size//2

    # mask off the edges where we can't form a complete patch
    mask[:delta, :] = 0
    mask[:, :delta] = 0
    mask[-delta:, :] = 0
    mask[:, -delta:] = 0

    # find the coords where the mask is True
    x, y = np.where(mask)
    n = len(x)

    # the max_count is at most the number of coords
    if not max_count:
        max_count = n
    else:
        max_count = min(max_count, n)

    # sample from the image where the mask is True
    if not random:
        idx = np.arange(max_count)
    else:
        idx = np.random.choice(n, max_count, replace=False)
    for i in idx:
        h, k = x[i], y[i]
        patch = ar[h-delta:h+delta, k-delta:k+delta]
        yield patch


def extract_patches(image, mask_p, n, pos_ratio=1, edge_ratio=1, bg_ratio=0.3):
    '''Samples labeled patches from an image given a positive mask.

    The negative class is sampled from an edge mask and a background mask,
    which are generated from the input image and positive mask.
    '''
    # generate the masks
    image = np.require(image)
    mask_p = np.require(mask_p)
    mask_e = edge_mask(mask_p)
    mask_b = background_mask(mask_p)

    # get patches for each mask
    p = list(extract_from_mask(image, mask_p, int(n * pos_ratio), random=True))
    e = list(extract_from_mask(image, mask_e, int(n * edge_ratio), random=True))
    b = list(extract_from_mask(image, mask_b, int(n * bg_ratio), random=True))

    # separate into positive and negative classes
    pos = p
    neg = e+b

    # if the classes are imbalanced, throw away some extras
    if len(neg) < len(pos):
        pos = pos[:len(neg)]

    return pos, neg


def create_cv(path='./data/epi', k=5, n=10000, **kwargs):
    '''Extract a training set of patches taken from all images in a directory.

    The dataset is folded for cross-validation by patient id.

    Kwargs are passed to `extract_patches`.
    '''
    logger.debug('creating cross validation folds')
    data_dir = get_data(path)
    masks = sorted(data_dir.glob('masks/*_mask.png'))
    images = [data_dir / f'{m.stem[:-5]}.tif' for m in masks]

    folds = [{'pos':[], 'neg':[]} for _ in range(k)]

    for i, (image_path, mask_path) in enumerate(zip(images, masks)):
        image = imread(image_path)
        mask_p = imread(mask_path, mask=True)
        meta = get_metadata(image_path)
        pos, neg = extract_patches(image, mask_p, n, **kwargs)
        f = hash(meta['patient']) % k
        folds[f]['pos'].extend(pos)
        folds[f]['neg'].extend(neg)

    for f in range(k):
        fold = folds[f]
        pos = len(fold['pos'])
        neg = len(fold['neg'])
        logger.debug(f'fold {f} has {pos}/{neg} positive/negative samples')

    return folds


def lazy_property(prop):
    '''A lazy, memoized version of the builtin `property` decorator.
    '''
    val = None
    def wrapper(self):
        nonlocal val
        if val is None:
            val = prop(self)
        return val
    return property(wrapper)


class Dataset(torch.utils.data.Dataset):
    '''A torch `Dataset` that combines positive and negative samples.
    '''

    def __init__(self, pos, neg):
        self._pos = pos
        self._neg = neg
        self._split = len(pos)

    def __len__(self):
        return len(self._pos) + len(self._neg)

    def __getitem__(self, i):
        i -= self._split
        if i < 0:
            x = self._pos[i]
            y = 1
        else:
            x = self._neg[i]
            y = 0
        x = np.transpose(x, (2,0,1))  # reorder to channels first for PyTorch
        return x, y


class EpitheliumSegmentation:
    '''A cross-validation loader for the epithelium segmentation dataset.
    '''

    def __init__(self, **kwargs):
        '''Create a dataloader.

        Kwargs:
            path (str):
                The path to the dataset.
            k (int):
                The number of cross-validation folds.
            n (int):
                A parameter to determine the number of
                patches drawn from each source image.
            size (int, default=64):
                The size of the image patches.
            pos_ratio (default=1):
                The dataset will contain `n * pos_ratio`
                positive patches per source image.
            edge_ratio (default=1):
                The dataset will contain `n * edge_ratio`
                negative edge patches per source image.
            bg_ratio (default=0.3):
                The dataset will contain `n * bg_ratio`
                negative background patches per source image.
        '''
        self._args = kwargs

    @lazy_property
    def datasets(self):
        '''The list of datasets, one for each fold.
        '''
        logger.info('loading epithelium dataset...')
        folds = create_cv(**self._args)
        datasets = np.array([Dataset(f['pos'], f['neg']) for f in folds])
        return datasets

    def load(self, fold):
        '''Get the datasets for a given fold.

        Returns:
            Returns three datasets: train, validation, and test
        '''
        k = len(self.datasets)
        assert 0 <= fold < k

        test = self.datasets[fold]
        validation = self.datasets[(fold + 1) % k]
        train = [ds for ds in self.datasets if ds != test and ds != validation]
        train = torch.utils.data.ConcatDataset(train)

        return train, validation, test
