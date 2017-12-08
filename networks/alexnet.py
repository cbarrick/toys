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

    See:
        The Caffe version of this network
            https://github.com/BVLC/caffe/blob/1.0/examples/cifar10/cifar10_full.prototxt
        The Janowczyk and Madabhushi version, without dropout
            https://github.com/choosehappy/public/blob/master/DL%20tutorial%20Code/common/BASE-alexnet_traing_32w_db.prototxt
        The Janowczyk and Madabhushi version, with dropout
            https://github.com/choosehappy/public/blob/master/DL%20tutorial%20Code/common/BASE-alexnet_traing_32w_dropout_db.prototxt
    '''

    def __init__(self, num_classes=10, shape=(3, 128, 128)):
        super().__init__()

        # The Caffe version of this network uses LRN layers,
        # but Janowczyk and Madabhushi do not.
        self.features = N.Sequential(
            N.Conv2d(shape[0], 32, kernel_size=5, stride=1, padding=2),
            N.MaxPool2d(kernel_size=3, stride=2, padding=1),
            N.ReLU(inplace=True),

            N.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            N.ReLU(inplace=True),
            N.AvgPool2d(kernel_size=3, stride=2, padding=1),

            N.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            N.ReLU(inplace=True),
            N.AvgPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Janowczyk and Madabhushi do NOT have an activation between the linear
        # layers. This is clearly a bug, since two linear layers reduce to a
        # single layer. The dropout version of the network _does_ have ReLUs
        # after _both_ layers. In that case, the final ReLU seems odd.
        n = int(np.ceil(shape[1] / 2 / 2 / 2))
        m = int(np.ceil(shape[2] / 2 / 2 / 2))
        self.classifier = N.Sequential(
            N.Linear(64*n*m, 64),
            N.Linear(64, num_classes),
        )

        self.reset()

    def reset(self):
        # The initialization scheme is taken from Janowczyk and Madabhushi.
        self.features[0].weight.data.normal_(std=0.001)
        self.features[0].bias.data.zero_()
        self.features[3].weight.data.normal_(std=0.01)
        self.features[3].bias.data.zero_()
        self.features[6].weight.data.normal_(std=0.01)
        self.features[6].bias.data.zero_()
        self.classifier[0].weight.data.normal_(std=0.1)
        self.classifier[0].bias.data.zero_()
        self.classifier[2].weight.data.normal_(std=0.1)
        self.classifier[2].bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
