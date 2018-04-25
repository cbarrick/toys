from typing import Sequence

import numpy as np

import torch
from torch import nn
from torch.autograd import Variable

import toys
from toys import current_context, parse_activation, parse_initializer


class FullyConnected(nn.Module):
    def __init__(self, input_shape, output_shape, **kwargs):
        '''TODO: Document

        Arguments:
            input_shape (int or Sequence[int]):
                The shape of a single input instance, i.e. without a batch
                dimension. A scaler ``n`` is interpreted as ``(n,)``.
            output_shape (int or Sequence[int]):
                The shape of a single output instance, i.e. without a batch
                dimension. A scaler ``n`` is interpreted as ``(n,)``.
            activation ([Variable] -> Variable or str or None):
                An activation function to apply after the linear operation. If
                a string is given, it is interpreted by `parse_activation`.
                Explicitly pass `None` for no activation. The default is taken
                from the current context, falling back to `None`.
            initializer ([Tensor] -> Tensor or str):
                An initializer function for the weights. If a string is given,
                it is interpreted by `parse_initializer`. The default is taken
                from the current context, falling back to `'kaiming_uniform'`.
            bias_initializer ([Tensor] -> Tensor or str):
                An initializer function for the bias. If a string is given,
                it is interpreted by `parse_initializer`. The default is taken
                from the current context, falling back to `'constant:val=0'`.
        '''
        ctx = current_context()

        if not isinstance(input_shape, Sequence):
            input_shape = (input_shape,)

        if not isinstance(output_shape, Sequence):
            output_shape = (output_shape,)

        actv = ctx.get('activation', None)
        actv = kwargs.pop('activation', actv)
        if isinstance(actv, str):
            actv = parse_activation(actv)

        init = ctx.get('initializer', 'kaiming_uniform')
        init = kwargs.pop('initializer', init)
        if isinstance(init, str):
            init = parse_initializer(init)

        bias_init = ctx.get('initializer', 'constant:val=0')
        bias_init = kwargs.pop('initializer', bias_init)
        if isinstance(bias_init, str):
            bias_init = parse_initializer(bias_init)

        input_flat = int(np.prod(input_shape))
        output_flat = int(np.prod(output_shape))
        linear = nn.Linear(input_flat, output_flat)

        linear.weight.data = init(linear.weight.data)
        linear.bias.data = bias_init(linear.bias.data)

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.input_flat = input_flat
        self.output_flat = output_flat
        self.linear = linear
        self.actv = actv

    def forward(self, x):
        x = x.contiguous()
        x = x.view(-1, self.input_flat)
        y = self.linear(x)
        y = y.view(-1, *self.output_shape)
        if self.actv:
            y = self.actv(y)
        return y
