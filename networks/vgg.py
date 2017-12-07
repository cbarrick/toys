import logging

import numpy as np

import torch
import torch.nn as N


logger = logging.getLogger(__name__)


class VggBlock2d(N.Module):
    def __init__(self, *chans):
        super().__init__()
        layers = []
        n = len(chans)
        for i in range(n-1):
            conv = N.Conv2d(chans[i], chans[i+1], kernel_size=3, stride=1, padding=2)
            relu = N.ReLU(inplace=True)
            layers += [conv, relu]
        layers += [N.MaxPool2d(kernel_size=2, stride=2)]
        self.layers = N.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class VggFrontend(N.Module):
    def __init__(self, *chans):
        super().__init__()
        layers = []
        n = len(chans)
        for i in range(n-1):
            full = N.Linear(chans[i], chans[i+1])
            relu = N.ReLU(inplace=True)
            layers += [full, relu]
        self.layers = N.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Vgg16(N.Module):
    def __init__(self, num_classes=10, shape=(3, 224, 224)):
        super().__init__()

        self.cnn = N.Sequential(
            VggBlock2d(shape[0], 64, 64),
            VggBlock2d(64, 128, 128),
            VggBlock2d(128, 256, 256, 256),
            VggBlock2d(256, 512, 512, 512),
            VggBlock2d(512, 512, 512, 512),
        )

        n = int(np.ceil(shape[1] / 2 / 2 / 2 / 2 / 2))
        m = int(np.ceil(shape[2] / 2 / 2 / 2 / 2 / 2))
        logger.debug(f'vgg expected shape {512*n*m}')
        self.frontend = VggFrontend(512*n*m, 4096, 4096, num_classes)

        self.reset()

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        logger.debug(f'vgg got shape {x.size(1)}')
        x = self.frontend(x)
        return x

    def reset(self):
        for m in self.modules():
            if isinstance(m, (N.Conv2d, N.Linear)):
                N.init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
