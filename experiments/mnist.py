#!/usr/bin/env python3
import argparse
import logging
from types import SimpleNamespace

import numpy as np
import sklearn.metrics

import torch
import torch.nn as N
import torch.optim as O

from datasets import MNIST
from datasets import FashionMNIST
from networks import AlexNet
from estimators import Classifier
from metrics import TruePositives, FalsePositives, TrueNegatives, FalseNegatives
from metrics import Accuracy, Precision, Recall, FScore


logger = logging.getLogger()


def seed(n):
    '''Seed the RNGs of stdlib, numpy, and torch.'''
    import random
    import numpy as np
    import torch
    random.seed(n)
    np.random.seed(n)
    torch.manual_seed(n)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(n)


def main(**kwargs):
    kwargs.setdefault('epochs', 600)
    kwargs.setdefault('learning_rate', 0.001)
    kwargs.setdefault('patience', None)
    kwargs.setdefault('batch_size', 128)
    kwargs.setdefault('cuda', None)
    kwargs.setdefault('dry_run', False)
    kwargs.setdefault('name', None)
    kwargs.setdefault('seed', 1337)
    kwargs.setdefault('verbose', 'WARN')
    kwargs.setdefault('tasks', ['mnist', 'fashion'])
    args = SimpleNamespace(**kwargs)

    logging.basicConfig(
        level=args.verbose,
        style='{',
        format='[{levelname:.4}][{asctime}][{name}:{lineno}] {msg}',
    )

    logger.debug('parameters of this experiment')
    for key, val in args.__dict__.items():
        logger.debug(f' {key:.15}: {val}')

    seed(args.seed)

    datasets = {
        'mnist': MNIST(),
        'fashion': FashionMNIST(),
    }

    if args.name is None:
        now = np.datetime64('now')
        args.name = f'exp-{now}'
        logger.info(f'experiment name not given, defaulting to {args.name}')

    # In some cases, we must move the network to it's cuda device before
    # constructing the optimizer. This is annoying, and this logic is
    # duplicated in the estimator class. Ideally, I'd like the estimator to
    # handle cuda allocation _after_ the optimizer has been constructed...
    net = AlexNet((1, 27, 27), ndim=10)
    if args.cuda is None:
        args.cuda = 0 if torch.cuda.is_available() else False
    if args.cuda is not False:
        net = net.cuda(args.cuda)

    for task in args.tasks:
        net.reset()
        opt = O.Adam(net.parameters(), lr=args.learning_rate)
        loss = N.CrossEntropyLoss()
        model = Classifier(net, opt, loss, name=args.name, cuda=args.cuda, dry_run=args.dry_run)

        data = datasets[task]
        train, test = data.load()

        print(f'-------- Fitting {task} --------')
        model.fit(train, epochs=args.epochs, patience=args.patience, batch_size=args.batch_size)
        print()

        print(f'-------- Scoring {task} --------')
        scores = {
            'accuracy': Accuracy(),
            'true positives': TruePositives(),
            'false positives': FalsePositives(),
            'true negatives': TrueNegatives(),
            'false negatives': FalseNegatives(),
            'precision': Precision(),
            'recall': Recall(),
            'f-score': FScore(),
        }
        for metric, criteria in scores.items():
            score = model.test(test, criteria, batch_size=args.batch_size)
            print(f'{metric:15}: {score}')
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        add_help=False,
        argument_default=argparse.SUPPRESS,
        formatter_class=argparse.RawTextHelpFormatter,
        description=(
            'Runs an experiment.\n'
            '\n'
            'Note that the experiment is intended to be executed from the root of the\n'
            'repository using `python -m`:\n'
            '\n'
            '  python -m experiments.mnist --cuda=0\n'
            '\n'
        ),
        epilog=(
            'Tasks:\n'
            '  mnist    The standard MNIST digit recognition dataset\n'
            '  fashion  A fashion dataset with the same dimensions and classes as MNIST\n'
        ),
    )

    group = parser.add_argument_group('Hyper-parameters')
    group.add_argument('-e', '--epochs', metavar='X', type=int)
    group.add_argument('-l', '--learning-rate', metavar='X', type=float)
    group.add_argument('-p', '--patience', metavar='X', type=int)

    group = parser.add_argument_group('Performance')
    group.add_argument('-b', '--batch-size', metavar='X', type=int)
    group.add_argument('-c', '--cuda', metavar='X', type=int)

    group = parser.add_argument_group('Debugging')
    group.add_argument('-d', '--dry-run', action='store_true')
    group.add_argument('-v', '--verbose', action='store_const', const='DEBUG')

    group = parser.add_argument_group('Other')
    group.add_argument('--seed')
    group.add_argument('--name', type=str)
    group.add_argument('--help', action='help')

    group = parser.add_argument_group('Positional')
    group.add_argument('tasks', metavar='TASK', nargs='*')

    args = parser.parse_args()
    main(**vars(args))
