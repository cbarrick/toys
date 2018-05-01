from typing import Sequence

import torch
from torch import nn


class MaxPool2d(nn.Module):
    def __init__(self, kernel_size, **kwargs):
        stride = kwargs.get('stride', kernel_size)
        padding = kwargs.get('padding', 0)
        dilation = kwargs.get('dilation', 1)
        return_indices = kwargs.get('return_indices', False)
        ceil_mode = kwargs.get('ceil_mode', False)

        # This layer accepts NHWC but torch expects NCHW. In the forward pass,
        # we transpose C and H. This puts H and W out of order (NCWH), so we
        # transpose the kernel rather than adding more transpose ops to forward.
        if isinstance(kernel_size, Sequence):
            (height, width) = kernel_size
            kernel_size = (width, height)
        if isinstance(stride, Sequence):
            (height, width) = stride
            stride = (width, height)

        self.pool = nn.MaxPool2d(kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            return_indices=return_indices, ceil_mode=ceil_mode)

    def forward(x):
        (*batch, height, width, channels) = x.shape
        x = x.view(-1, height, width, channels)
        x = x.transpose(-1, -3)
        x = self.pool(x)
        x = x.transpose(-3, -1)
        x = x.view(*batch, height, width, -1)
        return x
