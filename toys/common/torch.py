from typing import Any, Sequence, Union
import logging

import numpy as np

import torch
from torch.nn import Module

from .parsers import parse_dtype
from .estimator import Model


logger = logging.getLogger(__name__)


class TorchModel(Model):
    '''A wrapper around PyTorch modules.

    This wrapper extends `torch.Module` to accept scalars, numpy arrays, and
    torch tensors as input and to return numpy arrays as output.

    A TorchModel is aware of the number of dimensions expected for each input.
    If an input has fewer dimensions, it is extended with trival dimensions.

    Attributes:
        module (Module):
            The module being wrapped.
        dims (Sequence[int or None]):
            The number of dimensions expected of each input.
            A value of `None` means any shape is allowed.
    '''

    def __init__(self, module, *dims):
        '''Construct a TorchModel.

        Arguments:
            module (Module):
                The module being wrapped.
            dims (int or None):
                The number of dimensions required of each input. The number
                and order of dimensions must match those expected by the
                module. A value of `None` means any shape is allowed.
        '''
        self.module = module
        self.dims = dims

    def __getattr__(self name):
        '''Attribute access is delecated to the underlying module.
        '''
        return getattr(self.module, name)

    def __call__(self, *inputs):
        '''Evaluate the model on some inputs.
        '''
        with torch.no_grad():
            inputs = self._cast_inputs(*inputs)
            y = self.module(*inputs)
            y = y.numpy()
        return y

    def _cast_inputs(self, *inputs):
        '''Cast inputs to tensors of the expected dtype, device, and dimension.
        '''
        assert len(inputs) == len(self.dims)
        dtype = self.module.dtype
        device = self.module.device

        for x, n_dims in zip(inputs, self.dims):
            if np.isscalar(x): x = np.array(x)
            if isinstance(x, np.ndarray): x = torch.from_numpy(x)
            x = x.to(device, dtype)

            assert x.dim() <= n_dims
            for _ in range(x.dim(), n_dims):
                x.unsqueeze_(0)

            yield x
