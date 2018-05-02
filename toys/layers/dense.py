from typing import Sequence

import numpy as np

import torch
from torch import Tensor
from torch import nn

import toys
from toys.parsers import parse_activation, parse_initializer


class Dense(nn.Module):
    def __init__(self, in_shape, *shapes, **kwargs):
        '''Construct a fully connected layer.

        Arguments:
            in_shape (int or Sequence[int]):
                The shape of a single input instance, i.e. without a batch
                dimension. A scaler ``n`` is interpreted as a single dimension.
            shapes (int or Sequence[int]):
                The shape of a single output instance, i.e. without a batch
                dimension. A scaler ``n`` is interpreted as a single dimension.

        Keyword Arguments:
            bias (bool):
                If set to False, the layer will not learn an additive bias.
                Default: ``True``.
            activation ([Tensor] -> Tensor or str or None):
                An activation function to apply after the convolution.
                Default: ``None``.
            initializer ([Tensor] -> Tensor or str):
                An initializer function for the weights.
                Default: ``'kaiming_uniform'``.
            bias_initializer ([Tensor] -> Tensor or str):
                An initializer function for the bias.
                Default: ``'constant:val=0'``.
        '''
        assert 0 < len(shapes)
        out_shape = shapes[-1]

        bias = kwargs.get('bias', True)
        actv = kwargs.get('activation', None)
        init = kwargs.get('initializer', 'kaiming_uniform')
        bias_init = kwargs.get('bias_initializer', 'constant:val=0')

        if not isinstance(in_shape, Sequence):
            in_shape = (in_shape,)

        if not isinstance(out_shape, Sequence):
            out_shape = (out_shape,)

        if isinstance(actv, str):
            actv = parse_activation(actv)

        if isinstance(init, str):
            init = parse_initializer(init)

        if isinstance(bias_init, str):
            bias_init = parse_initializer(bias_init)

        layers = []
        prev = in_shape
        for shape in shapes:
            in_flat = int(np.prod(prev))
            out_flat = int(np.prod(shape))
            linear = nn.Linear(in_flat, out_flat, bias=bias)
            linear.weight = init(linear.weight)
            linear.bias = bias_init(linear.bias)
            layers.append(linear)
            prev = shape

        self._in_flat = int(np.prod(in_shape))
        self._out_shape = out_shape
        self.layers = nn.ModuleList(layers)
        self.actv = actv

    def forward(self, x):
        x = x.view(-1, self._in_flat)
        for layer in self.layers:
            x = self.linear(x)
            x = self.actv(x)
        x = x.view(-1, *self._out_shape)
        return x
