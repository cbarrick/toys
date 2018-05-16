from logging import getLogger

import numpy as np

import torch
from torch.nn import Module, Parameter

import toys
from toys.datasets.utils import DataLoader, Dataset
from toys.common import BaseEstimator, TorchModel
from toys.metrics import Mean
from toys.parsers import parse_dtype


logger = getLogger(__name__)


class FlatDataset(Dataset):
    def __init__(self, base):
        super().__init__()
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        *inputs, target = self.base[index]
        inputs = [x.reshape(-1) for x in inputs]
        inputs = np.concatenate(inputs)
        target = target.reshape(-1)
        return inputs, target


class LeastSquaresModule(Module):
    def __init__(self, weight, bias):
        super().__init__()
        self.weight = Parameter(weight, requires_grad=False)
        self.bias = Parameter(bias, requires_grad=False)

    def forward(self, *inputs):
        n = len(inputs[0])
        inputs = [x.view(n, -1) for x in inputs]
        inputs = torch.cat(inputs, dim=1)
        output = inputs @ self.weight + self.bias
        return output


class LeastSquares(BaseEstimator):
    def fit(self, *datasets, **kwargs):
        '''Trains a least squares model.

        Users should not call this method directly, but instead call the
        estimator as a function.

        Arguments:
            datasets (Dataset):
                The dataset to fit. At least one must be given. If multiple
                datasets are given, the columns from all datasets are combined
                into a single dataset. The rightmost column is treated as the
                target and the reset as inputs.

        Keyword Arguments:
            bias (bool):
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
            dry_run (bool):
                If true, break from loops early. Useful for debugging.
            **kwargs:
                Additional keyword arguments are forwarded to the `DataLoader`.

        Returns:
            model (TorchModel):
                A linear model minimizing the mean squared error. The model
                expects $n$ inputs where $n$ is the number of input columns
                in the training data.
        '''
        assert 0 < len(datasets)
        dataset = toys.zip(*datasets)
        dataset = FlatDataset(dataset)

        # Note that the keyword arg `bias` is bound to the local var `learn_bias`.
        # The local var called `bias` holds the actual learned bias.
        learn_bias = kwargs.setdefault('bias', True)
        batch_size = kwargs.setdefault('batch_size', len(dataset))
        max_epochs = kwargs.setdefault('max_epochs', 1)
        dtype = kwargs.setdefault('dtype', torch.get_default_dtype())
        dry_run = kwargs.setdefault('dry_run', False)

        dtype = parse_dtype(dtype)

        x, y = dataset[0]
        in_features = len(x)
        out_features = len(y)

        if batch_size < in_features or len(dataset) < in_features:
            logger.warning('least squares problem is under determined')
            logger.warning(f'consider using a batch_size of at least {in_features}')

        if learn_bias:
            x_mean = Mean(dim=0)
            y_mean = Mean(dim=0)
            for x, y in DataLoader(dataset, **kwargs):
                x_mean.accumulate(x)
                y_mean.accumulate(y)
            x_mean = x_mean.reduce()
            y_mean = y_mean.reduce()

        weight = Mean(dim=0)
        for i in range(max_epochs):
            for x, y in DataLoader(dataset, **kwargs):
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
        return TorchModel(mod, classifier=False, dtype=dtype)
