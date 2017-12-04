#!/usr/bin/env python3
import argparse
import logging
from types import SimpleNamespace

import numpy as np
import sklearn.metrics

import torch
import torch.autograd as A
import torch.nn as N
import torch.nn.functional as F
import torch.optim as O

from torchvision.datasets import MNIST

from models import AlexNet
from estimators import Classifier
from metrics import precision, recall, f_score


logger = logging.getLogger()


def main(**kwargs):
    kwargs.setdefault('data_size', 10000)
    kwargs.setdefault('epochs', 100)
    kwargs.setdefault('learning_rate', 0.001)
    kwargs.setdefault('patience', None)
    kwargs.setdefault('batch_size', 128)
    kwargs.setdefault('cuda', None)
    kwargs.setdefault('dry_run', False)
    kwargs.setdefault('name', 'mnist')
    kwargs.setdefault('verbose', 'WARN')
    kwargs.setdefault('net', 'alex')
    args = SimpleNamespace(**kwargs)

    logging.basicConfig(
        level=args.verbose,
        style='{',
        format='[{levelname:.4}][{asctime}][{name}:{lineno}] {msg}',
    )

    nets = {
        'alex': AlexNet(10),
    }

    metrics = {
        'precision': precision,
        'recall': recall,
        'f-score': f_score,
    }

    net = nets[args.net]
    opt = O.Adam(net.parameters(), lr=args.learning_rate)
    loss = N.CrossEntropyLoss()
    model = Classifier(net, opt, loss, name=args.name, cuda=args.cuda, dry_run=args.dry_run)

    print(f'-------- Training --------')
    train = MNIST('./data/mnist', train=True, download=True)
    model.fit(train, epochs=args.epochs, patience=args.patience, batch_size=args.batch_size)
    print()

    print(f'-------- Scoring --------')
    test = MNIST('./data/mnist', train=False, download=True)
    for metric, criteria in metrics.items():
        print(metric, end=': ', flush=True)
        z = model.test(test, criteria, batch_size=args.batch_size)
        print(z)
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run an experiment.',
        add_help=False,
        argument_default=argparse.SUPPRESS,
    )

    parser.add_argument('net', metavar='NET', nargs='?', help='The network to experiment with.')

    group = parser.add_argument_group('Hyper-parameters')
    group.add_argument('-n', '--data-size', metavar='N', type=int, help='The number of training samples is a function of N.')
    group.add_argument('-e', '--epochs', metavar='N', type=int, help='The maximum number of epochs per task.')
    group.add_argument('-l', '--learning-rate', metavar='N', type=float, help='The learning rate.')
    group.add_argument('-p', '--patience', metavar='N', type=int, help='Higher patience may help avoid local minima.')

    group = parser.add_argument_group('Performance')
    group.add_argument('-b', '--batch-size', metavar='N', type=int, help='The batch size.')
    group.add_argument('-c', '--cuda', metavar='N', type=int, help='Use the Nth cuda device.')

    group = parser.add_argument_group('Debugging')
    group.add_argument('-d', '--dry-run', action='store_true', help='Do a dry run to check for errors.')
    group.add_argument('-v', '--verbose', action='store_const', const='DEBUG', help='Turn on debug logging.')

    group = parser.add_argument_group('Other')
    group.add_argument('--name', type=str, help='Sets a name for the experiment.')
    group.add_argument('--help', action='help', help='Show this help message and exit.')

    args = parser.parse_args()
    main(**vars(args))
