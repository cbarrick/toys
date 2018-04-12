import logging
import numpy as np

import torch
from torch.autograd import Variable


logger = logging.getLogger(__name__)


# A mapping from dtype names to PyTorch tensor classes.
# Dtype names can be conventional, e.g. 'float' and 'double',
# or explicit, e.g. 'float32' and 'float64'.
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


class TorchModel:
    '''A wrapper around PyTorch modules.

    The primary purpose of this class is to (a) support numpy and scalar
    inputs and (b) to cast outputs to numpy arrays.

    Attributes:
        module: The module being wrapped.
        dtype: The data type being used.
        is_cuda: True if the module has been moved to a CUDA device.
    '''

    def __init__(self, module, dtype):
        '''Construct a TorchModel.

        Args:
            module:
                The PyTorch module being wrapped. The module is cast to the
                given dtype and moved to the CPU.
            dtype:
                The PyTorch data type to operate on, i.e. a Tensor class.
                The dtype may be given as a string like 'double' or 'float64'.
                The module and all inputs are cast to this type.
        '''
        if isinstance(dtype, str):
            dtype = TORCH_DTYPES[dtype]
        module = module.type(dtype).cpu()
        self.module = module
        self.dtype = dtype
        self.is_cuda = False

    def cuda(self, device=None):
        '''Move the module to a CUDA device.

        Args:
            device: The device to use. Defaults to the first available.

        Returns:
            self
        '''
        self.module = self.module.cuda(device)
        self.is_cuda = True
        return self

    def cpu(self):
        '''Move the module to the CPU.

        Returns:
            self
        '''
        self.module = self.module.cpu()
        self.is_cuda = False
        return self

    def __call__(self, *inputs):
        '''Evaluate the model on some inputs.

        The inputs may be PyTorch Tensors, numpy arrays, or scalars.
        The output will be a numpy array.

        Args:
            inputs: Passed to the underlying module.

        Returns:
            The output of the module.

        TODO:
            Support out-of-core datasets (dask arrays).
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
