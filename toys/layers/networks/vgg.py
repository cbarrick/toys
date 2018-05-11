import torch
from torch import nn

import toys
from toys.layers import Conv2d
from toys.layers import Dense
from toys.layers import MaxPool2d


def parse_args(kwargs):
    common_args = {
        'bias': True,
        'activation': 'relu',
        'initializer': 'kaiming_uniform',
        'bias_initializer': 'constant:val=0',
        **kwargs,
    }

    conv_args = {
        'kernel_size': 3,
        'padding': 1,
        'pooling': lambda: MaxPool2d(2),
        **common_args,
    }

    dense_args = {
        **common_args,
    }

    output_args = {
        **common_args,
        'activation': None,
        'initializer': 'xavier_uniform',
    }

    return conv_args, dense_args, output_args


class _VGGBase(nn.Module):
    def __init__(self, cnn, dense, output):
        super().__init__()
        self.cnn = cnn
        self.dense = dense
        self.output = output

    def forward(self, x):
        x = self.cnn(x)
        x = self.dense(x)
        x = self.output(x)
        return x


class VGG11(_VGGBase):
    def __init__(self, in_shape, out_shape, **kwargs):
        (*batch, height, width, in_channels) = in_shape
        transition_shape = (*batch, height//2**5, width//2**5, 512)

        Conv = kwargs.setdefault('conv', Conv2d)
        conv_args, dense_args, output_args = parse_args(kwargs)

        # Max pooling is automatically applied at the end of each conv layer.
        # Use the `pooling` keyword argument to override.
        cnn = nn.Sequential(
            Conv(in_channels, 64, **conv_args),
            Conv(64, 128, **conv_args),
            Conv(128, 256, 256, **conv_args),
            Conv(256, 512, 512, **conv_args),
            Conv(512, 512, 512, **conv_args),
        )

        dense = Dense(transition_shape, 4096, 4096, **dense_args)
        output = Dense(4096, out_shape, **output_args)

        super().__init__(cnn, dense, output)


class VGG13(_VGGBase):
    def __init__(self, in_shape, out_shape, **kwargs):
        (*batch, height, width, in_channels) = in_shape
        transition_shape = (*batch, height//2**5, width//2**5, 512)

        Conv = kwargs.setdefault('conv', Conv2d)
        conv_args, dense_args, output_args = parse_args(kwargs)

        # Max pooling is automatically applied at the end of each conv layer.
        # Use the `pooling` keyword argument to override.
        cnn = nn.Sequential(
            Conv(in_channels, 64, 64, **conv_args),
            Conv(64, 128, 128, **conv_args),
            Conv(128, 256, 256, **conv_args),
            Conv(256, 512, 512, **conv_args),
            Conv(512, 512, 512, **conv_args),
        )

        dense = Dense(transition_shape, 4096, 4096, **dense_args)
        output = Dense(4096, out_shape, **output_args)

        super().__init__(cnn, dense, output)


class VGG16(_VGGBase):
    def __init__(self, in_shape, out_shape, **kwargs):
        (*batch, height, width, in_channels) = in_shape
        transition_shape = (*batch, height//2**5, width//2**5, 512)

        Conv = kwargs.setdefault('conv', Conv2d)
        conv_args, dense_args, output_args = parse_args(kwargs)

        # Max pooling is automatically applied at the end of each conv layer.
        # Use the `pooling` keyword argument to override.
        cnn = nn.Sequential(
            Conv(in_channels, 64, 64, **conv_args),
            Conv(64, 128, 128, **conv_args),
            Conv(128, 256, 256, 256, **conv_args),
            Conv(256, 512, 512, 512 **conv_args),
            Conv(512, 512, 512, 512, **conv_args),
        )

        dense = Dense(transition_shape, 4096, 4096, **dense_args)
        output = Dense(4096, out_shape, **output_args)

        super().__init__(cnn, dense, output)


class VGG19(_VGGBase):
    def __init__(self, in_shape, out_shape, **kwargs):
        (*batch, height, width, in_channels) = in_shape
        transition_shape = (*batch, height//2**5, width//2**5, 512)

        Conv = kwargs.setdefault('conv', Conv2d)
        conv_args, dense_args, output_args = parse_args(kwargs)

        # Max pooling is automatically applied at the end of each conv layer.
        # Use the `pooling` keyword argument to override.
        cnn = nn.Sequential(
            Conv(in_channels, 64, 64, **conv_args),
            Conv(64, 128, 128, **conv_args),
            Conv(128, 256, 256, 256, 256, **conv_args),
            Conv(256, 512, 512, 512, 512, **conv_args),
            Conv(512, 512, 512, 512, 512, **conv_args),
        )

        dense = Dense(transition_shape, 4096, 4096, **dense_args)
        output = Dense(4096, out_shape, **output_args)

        super().__init__(cnn, dense, output)
