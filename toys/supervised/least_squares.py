from logging import getLogger

import numpy as np

import torch
from torch.nn import Module, Parameter

import toys
from toys.common import BaseEstimator, TorchModel
from toys.data import Dataset
from toys.metrics import Mean
from toys.parsers import parse_dtype


logger = getLogger(__name__)


class LeastSquaresModule(Module):
    def __init__(self, weight, bias):
        super().__init__()
        self.weight = Parameter(weight, requires_grad=False)
        self.bias = Parameter(bias, requires_grad=False)

    def forward(self, *args):
        n = len(args[0])
        args = [x.view(n, -1) for x in args]
        args = torch.cat(args, dim=1)
        output = args @ self.weight + self.bias
        return output


class LeastSquares(BaseEstimator):
    def fit(self, *datasets, learn_bias=True, batch_size=None, max_epochs=1, dtype=None,
            underdetermined=False, dry_run=False):
        '''Trains a least squares model.

        Users should not call this method directly, but instead call the
        estimator as a function.

        .. todo::
            :class:`LeastSquares` does not yet support CUDA.

        Arguments:
            datasets (Dataset):
                The datasets to fit. If more than one are given, they are
                combined using `toys.zip`. The target is taken from the last
                column.

        Keyword Arguments:
            learn_bias (bool):
                If true (the default), learn an additive bias/intercept term.
                If false, the intercept is always 0.
            batch_size (int):
                The number of samples in each batch. This should be greater
                than the number of input features. The default is to read the
                entire dataset in a single batch.
            max_epochs (int):
                The number of times to iterate over the dataset.
            dtype (str or torch.dtype):
                Cast the module to this data type. This can be a PyTorch dtype
                object, a conventional name like 'float' and 'double', or an
                explicit name like 'float32' and 'float64'. The default is
                determined by `torch.get_default_dtype` and may be set with
                `torch.set_default_dtype`.
            underdetermined (bool):
                The estimator issues a warning if the problem is
                underdetermined, i.e. the batch size is less than the number
                of features. Set this to true to ignore the warning.
            dry_run (bool):
                If true, break from loops early. Useful for debugging.

        Returns:
            model (TorchModel):
                A linear model minimizing the mean squared error. The model
                expects $n$ inputs where $n$ is the number of feature columns
                in the training data.
        '''
        dataset = toys.zip(*datasets)
        dataset = toys.flatten(dataset, supervised=True)

        batch_size = batch_size or len(dataset)
        batch_size = min(batch_size, len(dataset))

        dtype = dtype or torch.get_default_dtype()
        dtype = parse_dtype(dtype)

        x, y = dataset[0]
        in_features = len(x)
        out_features = len(y)

        if not underdetermined and batch_size < in_features:
            logger.warning('least squares problem is underdetermined')
            logger.warning(f'  this means that the batch size is less than the number of features')
            logger.warning(f'  batch_size={batch_size}, features={in_features}')
            logger.warning(f'  set underdetermined=True to disable this warning')

        if learn_bias:
            x_mean = Mean(dim=0)
            y_mean = Mean(dim=0)
            for x, y in toys.batches(dataset, batch_size):
                x_mean.accumulate(x)
                y_mean.accumulate(y)
            x_mean = x_mean.reduce()
            y_mean = y_mean.reduce()

        weight = Mean(dim=0)
        for i in range(max_epochs):
            for x, y in toys.batches(dataset, batch_size):
                if learn_bias:
                    x -= x_mean
                    y -= y_mean

                # gels is the LAPACK routine for least squares.
                # https://pytorch.org/docs/stable/torch.html#torch.gels
                # https://software.intel.com/en-us/mkl-developer-reference-c-gels
                batch_weight, _ = torch.gels(y, x)
                batch_weight = batch_weight[:in_features]
                assert batch_weight.shape == (in_features, out_features)

                # We duplicate the solution for this batch to weight it by the
                # batch size. This has no memory overhead, see `Tensor.expand`.
                batch_weight = batch_weight.unsqueeze(0)
                batch_weight = batch_weight.expand(len(x), in_features, out_features)
                weight.accumulate(batch_weight)

                if dry_run:
                    break

        weight = weight.reduce()
        bias = y_mean - x_mean @ weight if learn_bias else 0

        mod = LeastSquaresModule(weight, bias)
        mod = TorchModel(mod, device_ids=[], dtype=dtype)
        mod = mod.eval()
        return mod
