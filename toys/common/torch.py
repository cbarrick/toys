from typing import Tuple

import numpy as np

import torch
from torch.nn import Module

import toys
from toys.estimator import Model


class TorchModel(Model):
    '''A wrapper around PyTorch modules.

    This wrapper extends `torch.Module` to accept scalars, numpy arrays, and
    torch tensors as input and to return numpy arrays as output.

    A `TorchModel` is aware of the number of dimensions expected for each
    input. If an input has fewer dimensions, trivial axes are added.

    A `TorchModel` is NOT a `torch.Module`. Gradients are never computed.

    Attributes:
        module (Module):
            The module being wrapped.
        dims (Tuple[int or None] or None):
            The number of dimensions required of each input. If present,
            the number and order of dimensions must match the number and
            order of inputs expected by the module. A value of ``None``
            means any shape is allowed for the corresponding input. If not
            present, the number and shape of inputs is unconstrained.
    '''

    def __init__(self, module, *dims):
        '''Construct a `TorchModel`.

        Arguments:
            module (Module):
                The module being wrapped.
            dims (int or None):
                The number of dimensions required of each input. If present,
                the number and order of dimensions must match the number and
                order of inputs expected by the module. A value of ``None``
                means any shape is allowed for the corresponding input. If not
                present, the number and shape of inputs is unconstrained.
        '''
        self.module = module
        self.dims = dims or None

    def __getattr__(self, name):
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
        assert self.dims is None or len(inputs) == len(self.dims)
        dtype = self.module.dtype
        device = self.module.device

        for i, x in enumerate(inputs):
            if np.isscalar(x):
                x = np.array(x)

            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)

            x = x.to(device, dtype)

            if self.dims and self.dims[i]:
                assert x.dim() <= n_dims
                for _ in range(x.dim(), n_dims):
                    x = x.unsqueeze_(0)

            yield x
