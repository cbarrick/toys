import logging
import re
import requests
import tarfile
from pathlib import Path

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn.naive_bayes

import torch
import torch.utils.data


logger = logging.getLogger(__name__)


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


def download(url, dst):
    '''Download a tgz file and extract it to `dst`.
    '''
    url = str(url)
    dst = Path(dst)
    print(f'downloading {url}')
    r = requests.get(url, stream=True)
    with tarfile.open(mode='r:gz', fileobj=r.raw) as tar:
        print(f'extracting to {dst}')
        dst.mkdir(parents=True, exist_ok=True)
        tar.extractall(dst)


def get_data(path='./data/tubule', force_download=False):
    '''Get the dataset.

    Returns:
        Returns a list of dicts, one for each image.

    Keys:
        path: The path to the image.
        image: The image as a numpy array.
        mask: The mask of the positive class.
        benign: True if the tubule is benign, False if malignant.
        patient: An ID for the patient.
        id: An ID for this image.
    '''
    url = 'http://andrewjanowczyk.com/wp-static/tubule.tgz'
    path = Path(path)

    # We assume the download has been completed successfully if the destination already exists.
    # This is not very general, so use `force_download` if you're unsure about the correctness of the data.
    if not path.exists() or force_download:
        download(url, path)

    # Get the paths to images and masks.
    # These will align with each other if the data is correct.
    images = sorted(p for p in path.iterdir() if re.search('[0-9]+-[0-9]+-[0-9]+-[0-9]+.bmp', str(p)))
    masks =  sorted(p for p in path.iterdir() if re.search('[0-9]+-[0-9]+-[0-9]+-[0-9]+_anno.bmp', str(p)))
    assert len(masks) == len(images)

    # Get the list of benign cases.
    # (We could get the list of malignant cases instead.)
    with open(path / 'benign.txt') as f:
        benign = f.read().split('\n')

    # Loop over the paths to load the images and extract the metadata.
    # The first three numbers uniquely identify the patient.
    # The last number is the image ID for that patient.
    data = []
    for image, mask in zip(images, masks):
        match = re.search('([0-9]+-[0-9]+-[0-9]+)-([0-9]+)', str(image))
        data.append({
            'path': image,
            'image': imread(image),
            'mask': imread(mask, mask=True),
            'benign': image.name in benign,
            'patient': match[1],
            'id': int(match[2]),
        })

    return data


def soft_masks(image, mask):
    '''Compute probabalistic masks from a binary positive mask.

    This function computes a pair of masks, for the positive and negative
    classes, such that each value is the inverse confidence of a naive bayes
    classifier for that pixel. In other words, the larger the pixel value,
    the worse naive bayes is at predicting its class.

    These soft masks are used to sample difficult patches for training.
    '''
    nb = sklearn.naive_bayes.GaussianNB()
    x = image.reshape((-1, 3))
    y = mask.reshape((-1,))
    nb.fit(x, y)

    m = nb.predict_proba(x)
    m = m.reshape((*mask.shape, 2))

    pos = m[..., 1]
    pos = mask - pos
    pos = pos.clip(0)

    neg = m[..., 0]
    neg = np.logical_not(mask) - neg
    neg = neg.clip(0)

    return pos, neg


def sample_patch_idx(mask, max_count=None):
    '''Sample pixel indices from a mask.

    The higher the value of the pixel in the mask,
    the more likely it is to be sampled.
    '''
    # the number of samples taken should be no more
    # than the number of labeled pixels.
    k = min(
        max_count or float('inf'),
        (mask > 0).sum(),
    )

    # sample from the image using the mask as the probabilities
    mask = mask / mask.sum()
    n = np.prod(mask.shape)
    idx = np.random.choice(n, size=k, replace=False, p=mask.flat)
    idx = np.unravel_index(idx, mask.shape)
    return idx


def extract_from_mask(image, mask, max_count=None, size=128):
    '''Sample patches from a mask.

    The higher the value of the pixel in the mask, the more likely a patch is
    to be sampled with that pixel in the center.
    '''
    mask = np.copy(mask)
    width, height = mask.shape
    delta = size//2

    # mask off the edges where we can't form a complete patch
    mask[:delta, :] = 0
    mask[:, :delta] = 0
    mask[-delta:, :] = 0
    mask[:, -delta:] = 0

    # loop over the patches
    idx = sample_patch_idx(mask, max_count)
    for i, j in zip(*idx):
        yield image[i-delta:i+delta, j-delta:j+delta]


def extract_patches(image, mask, n, pos_ratio=1, bg_ratio=1):
    '''Samples labeled patches from an image given a positive mask.

    A patch is more likely to be sampled if a naive bayes classifier is worse
    as accurately classifying its center pixel.
    '''
    # generate the masks
    mask_p, mask_b = soft_masks(image, mask)

    # get patches for each mask
    pos = list(extract_from_mask(image, mask_p, max_count=int(n * pos_ratio)))
    neg = list(extract_from_mask(image, mask_b, max_count=int(n * bg_ratio)))

    # if the classes are imbalanced, throw away some extras
    if len(neg) < len(pos):
        pos = pos[:len(neg)]

    return pos, neg


def create_cv(path='./data/tubule', k=5, n=10000, **kwargs):
    '''Extract a training set of patches taken from all images in a directory.

    The dataset is folded for cross-validation by patient id.

    Kwargs are passed to `extract_patches`.
    '''
    logger.debug('creating cross validation folds')
    data = get_data(path)
    folds = [{'pos':[], 'neg':[]} for _ in range(k)]

    for i, x in enumerate(data):
        pos, neg = extract_patches(x['image'], x['mask'], n, **kwargs)
        f = hash(x['patient']) % k
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


class TubuleSegmentation:
    '''A cross-validation loader for the nuclei segmentation dataset.
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
            bg_ratio (default=0.3):
                The dataset will contain `n * bg_ratio`
                negative background patches per source image.
        '''
        self._args = kwargs

    @lazy_property
    def datasets(self):
        '''The list of datasets, one for each fold.
        '''
        logger.info('loading nuclei dataset...')
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
