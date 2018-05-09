from functools import partial
from typing import Callable, Sequence

import numpy as np

import torch
from torch import Tensor
from torch import nn

import toys
from toys.parsers import parse_activation, parse_initializer


class Conv2d(nn.Module):
    def __init__(self, in_channels, *channels, **kwargs):
        '''Construct a 2D convolution layer.

        Arguments:
            in_channels (int):
                The shape of feature channels in each inputs.
            channels (int):
                The number of activation maps in each layer.

        Keyword Arguments:
            kernel_size (int or Sequence[int]):
                Size of the convolving kernel. Default: 3.
            stride (float or int or Sequence[int]):
                Stride of the convolution. Default: 1.
            padding (int or Sequence[int]):
                Zero-padding added to both sides of the input. Default: 0.
            output_padding (int or Sequence[int]):
                Additional size added to one side of each dimension in the
                output shape, when using fractional stride. Default: 0.
            dilation (int or Sequence[int]):
                Spacing between kernel elements. Default: 1.
            groups (int):
                Number of blocked connections from input channels to output
                channels. Default: 1.
            bias (bool):
                If set to False, the layer will not learn an additive bias.
                Default: ``True``.
            pooling (Callable or None):
                A constructor for a pooling layer to apply after all
                convolutions. Default: None.
                **TODO**: Accept string values; requires extending `toys.parsers`.
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
        super().__init__()

        kernel_size = kwargs.get('kernel_size', 3)
        stride = kwargs.get('stride', 1)
        padding = kwargs.get('padding', 0)
        output_padding = kwargs.get('output_padding', 0)
        dilation = kwargs.get('dilation', 1)
        groups = kwargs.get('groups', 1)
        bias = kwargs.get('bias', True)
        pooling = kwargs.get('pooling', None)
        actv = kwargs.get('activation', None)
        init = kwargs.get('initializer', 'kaiming_uniform')
        bias_init = kwargs.get('bias_initializer', 'constant:val=0')

        actv = parse_activation(actv)
        init = parse_initializer(init)
        bias_init = parse_initializer(bias_init)

        assert 0 < len(channels)
        assert 0 < stride

        if 0 < stride < 1:
            stride = int(1/stride)
            Conv2d = partial(nn.ConvTranspose2d, output_padding=output_padding)
        else:
            assert output_padding == 0
            Conv2d = nn.Conv2d

        # TODO: create a parser for pooling arguments
        if pooling is None:
            pooling_layer = lambda x: x
        else:
            pooling_layer = pooling()

        layers = []
        prev = in_channels
        for c in channels:
            conv = Conv2d(prev, c,
                kernel_size=kernel_size, stride=stride, padding=padding,
                dilation=dilation, groups=groups, bias=bias)
            conv.weight = init(conv.weight)
            conv.bias = bias_init(conv.bias)
            layers.append(conv)
            prev = c

        self.layers = nn.ModuleList(layers)
        self.actv = actv
        self.pooling = pooling_layer

    def forward(self, x):
        (*batch, height, width, channels) = x.shape
        x = x.view(-1, height, width, channels)
        x = torch.einsum('nhwc->nchw', [x])
        for layer in self.layers:
            x = layer(x)
            x = self.actv(x)
        x = torch.einsum('nchw->nhwc', [x])
        x = x.view(*batch, height, width, -1)
        x = self.pooling(x)
        return x
