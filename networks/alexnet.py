import numpy as np

import torch
import torch.nn as N


class LRN(N.Module):
    '''A local response normalization layer.

    Written by @jiecaoyu on GitHub:
    https://github.com/pytorch/pytorch/issues/653
    '''
    def __init__(self, local_size=1, alpha=1.0, beta=0.75, cross_channel=False):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.cross_channel = cross_channel

        if self.cross_channel:
            self.average = N.AvgPool3d(
                kernel_size=(local_size, 1, 1),
                stride=1,
                padding=(int((local_size-1.0)/2), 0, 0),
            )
        else:
            self.average=N.AvgPool2d(
                kernel_size=local_size,
                stride=1,
                padding=int((local_size-1.0)/2),
            )

    def forward(self, x):
        if self.cross_channel:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        x = x.div(div)
        return x


class AlexNet(N.Module):
    '''An AlexNet-like model based on the CIFAR-10 variant of AlexNet in Caffe.

    See https://github.com/BVLC/caffe/blob/1.0/examples/cifar10/cifar10_full.prototxt
    '''

    def __init__(self, num_classes=10, shape=(3, 64, 64)):
        super().__init__()

        self.features = N.Sequential(
            N.Conv2d(shape[0], 32, kernel_size=5, stride=1, padding=2),
            N.MaxPool2d(kernel_size=3, stride=2, padding=1),
            N.ReLU(inplace=True),
            LRN(3, alpha=5e-5, beta=0.75, cross_channel=False),

            N.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            N.ReLU(inplace=True),
            N.AvgPool2d(kernel_size=3, stride=2, padding=1),
            LRN(3, alpha=5e-5, beta=0.75, cross_channel=False),

            N.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            N.ReLU(inplace=True),
            N.AvgPool2d(kernel_size=3, stride=2, padding=1),
        )

        n = int(np.ceil(shape[1] / 2 / 2 / 2))
        m = int(np.ceil(shape[2] / 2 / 2 / 2))
        self.classifier = N.Sequential(
            N.Linear(64*n*m, 64),
            N.ReLU(inplace=True),
            N.Dropout(),
            N.Linear(64, num_classes),
        )

        self.reset()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def reset(self):
        for m in self.modules():
            if isinstance(m, (N.Conv2d, N.Linear)):
                N.init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
