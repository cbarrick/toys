import numpy as np

import torch
import torch.nn as N


class VGG16(N.Module):

    def __init__(self, num_classes=10, shape=(3, 128, 128)):
        super().__init__()

        self.features = N.Sequential(
            N.Conv2d(shape[0], 64, kernel_size=3, stride=1, padding=1),
            N.ReLU(inplace=True),
            N.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            N.ReLU(inplace=True),
            N.MaxPool2d(kernel_size=2, stride=2),

            N.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            N.ReLU(inplace=True),
            N.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            N.ReLU(inplace=True),
            N.MaxPool2d(kernel_size=2, stride=2),

            N.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            N.ReLU(inplace=True),
            N.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            N.ReLU(inplace=True),
            N.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            N.ReLU(inplace=True),
            N.MaxPool2d(kernel_size=2, stride=2),

            N.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            N.ReLU(inplace=True),
            N.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            N.ReLU(inplace=True),
            N.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            N.ReLU(inplace=True),
            N.MaxPool2d(kernel_size=2, stride=2),

            N.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            N.ReLU(inplace=True),
            N.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            N.ReLU(inplace=True),
            N.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            N.ReLU(inplace=True),
            N.MaxPool2d(kernel_size=2, stride=2),
        )

        n = int(np.ceil(shape[1] / 2 / 2 / 2 / 2 / 2))
        m = int(np.ceil(shape[2] / 2 / 2 / 2 / 2 / 2))
        self.classifier = N.Sequential(
            N.Linear(512*n*m, 4096),
            N.ReLU(inplace=True),
            N.Linear(4096, 4096),
            N.ReLU(inplace=True),
            N.Linear(4096, num_classes),
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
