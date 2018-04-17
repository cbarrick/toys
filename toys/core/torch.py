from typing import Any, Sequence, Union
import logging

import numpy as np

import torch
from torch.autograd import Variable
from torch.nn import Module

from .estimator import Model


logger = logging.getLogger(__name__)


#: A common supertype for all torch Tensor classes.
TorchDtype = Union[
    # CPU tensors
    torch.FloatTensor,
    torch.DoubleTensor,
    torch.HalfTensor,
    torch.ByteTensor,
    torch.CharTensor,
    torch.ShortTensor,
    torch.IntTensor,
    torch.LongTensor,

    # CUDA tensors
    torch.cuda.FloatTensor,
    torch.cuda.DoubleTensor,
    torch.cuda.HalfTensor,
    torch.cuda.ByteTensor,
    torch.cuda.CharTensor,
    torch.cuda.ShortTensor,
    torch.cuda.IntTensor,
    torch.cuda.LongTensor,
]


#: A mapping from dtype names to PyTorch tensor classes.
#: Dtype names can be conventional, e.g. 'float' and 'double',
#: or explicit, e.g. 'float32' and 'float64'.
TORCH_DTYPES = {
    # Conventional names
    'float': torch.FloatTensor,
    'double': torch.DoubleTensor,
    'half': torch.HalfTensor,
    'byte': torch.ByteTensor,
    'char': torch.CharTensor,
    'short': torch.ShortTensor,
    'int': torch.IntTensor,
    'long': torch.LongTensor,

    # Explicit names
    'float32': torch.FloatTensor,
    'float64': torch.DoubleTensor,
    'float16': torch.HalfTensor,
    'uint8': torch.ByteTensor,
    'int8': torch.CharTensor,
    'int16': torch.ShortTensor,
    'int32': torch.IntTensor,
    'int64': torch.LongTensor,
}


def torch_dtype(dtype):
    '''Casts dtype to a PyTorch tensor class.

    The input may be a conventional name, like 'float' and 'double', or an
    explicit name like 'float32' or 'float64'. If the input is a known tensor
    class, it is returned as-is.

    Args:
        dtype (str or TorchDtype):
            A conventional name, explicit name, or known tensor class.

    Returns:
        cls (TorchDtype):
            The tensor class corresponding to `dtype`.
    '''
    try:
        return TORCH_DTYPES[dtype]
    except KeyError:
        if dtype not in TORCH_DTYPES.values():
            if isinstance(dtype, str):
                raise ValueError(f'unknown torch dtype {dtype}')
            else:
                raise TypeError(f'expected str or a tensor class, found {type(dtype)}')
        return dtype


class TorchModel(Model):
    '''A wrapper around PyTorch modules.

    The primary purpose of this class is to (a) support numpy and scalar
    inputs and (b) to cast outputs to numpy arrays.

    Attributes:
        module (Module):
            The module being wrapped.
        dtype (TorchDtype):
            The data type being used.
        is_cuda (bool):
            True if the module has been moved to a CUDA device.
    '''

    def __init__(self, module, dtype):
        '''Construct a TorchModel.

        Args:
            module (Module):
                The PyTorch module being wrapped. The module is cast to the
                given dtype and moved to the CPU.
            dtype (str or TorchDtype):
                The PyTorch data type to operate on, i.e. a Tensor class.
                The dtype may be given as a string like 'double' or 'float64'.
                The module and all inputs are cast to this type.
        '''
        dtype = torch_dtype(dtype)
        module = module.type(dtype).cpu()
        self.module = module
        self.dtype = dtype
        self.is_cuda = False

    def cuda(self, device=None):
        '''Move the module to a CUDA device.

        Args:
            device (int or None):
                The device to use. Defaults to the first available.

        Returns:
            self (TorchModel)
        '''
        self.module = self.module.cuda(device)
        self.is_cuda = True
        return self

    def cpu(self):
        '''Move the module to the CPU.

        Returns:
            self (TorchModel)
        '''
        self.module = self.module.cpu()
        self.is_cuda = False
        return self

    def __call__(self, *inputs):
        '''Evaluate the model on some inputs.

        The inputs may be PyTorch Tensors, numpy arrays, or scalars.
        The output will be a numpy array.

        Args:
            inputs (Sequence):
                Passed to the underlying module.

        Returns:
            output (Any):
                The output of the module.
        '''
        # Cast all inputs to the proper Variable type.
        def cast_to_tensor(x):
            if np.isscalar(x): x = [[x]]
            x = torch.Tensor(x).type(self.dtype)
            if self.is_cuda: x = x.cuda(async=True)
            x = Variable(x, volatile=True)
            return x

        # Do the prediction.
        inputs = (cast_to_tensor(x) for x in inputs)
        y = self.module(*inputs)

        # Cast back to numpy.
        y = y.data.cpu().numpy()
        return y
