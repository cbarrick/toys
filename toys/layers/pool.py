from typing import Sequence

import torch
from torch import nn


class MaxPool2d(nn.Module):
    def __init__(self, kernel_size, **kwargs):
        super().__init__()
        stride = kwargs.setdefault('stride', kernel_size)
        padding = kwargs.setdefault('padding', 0)
        dilation = kwargs.setdefault('dilation', 1)
        return_indices = kwargs.setdefault('return_indices', False)
        ceil_mode = kwargs.setdefault('ceil_mode', False)

        self.pool = nn.MaxPool2d(kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            return_indices=return_indices, ceil_mode=ceil_mode)

    def forward(self, x):
        (*batch, height, width, channels) = x.shape
        x = x.view(-1, height, width, channels)
        x = torch.einsum('nhwc->nchw', [x])
        x = self.pool(x)
        x = torch.einsum('nchw->nhwc', [x])
        (_, new_height, new_width, _) = x.shape
        x = x.contiguous()
        x = x.view(*batch, new_height, new_width, channels)
        return x
