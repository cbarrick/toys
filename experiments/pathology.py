#!/usr/bin/env python3
import argparse
import logging
from types import SimpleNamespace

import numpy as np
import sklearn.metrics

import torch
import torch.nn as N
import torch.optim as O

from datasets.pathology import NucleiSegmentation
from datasets.pathology import EpitheliumSegmentation
from datasets.pathology import TubuleSegmentation
from networks import AlexNet
from networks import Vgg16
from estimators import Classifier
from metrics import precision, recall, f_score


logger = logging.getLogger()


def seed(n):
    '''Seed the RNGs of stdlib, numpy, and torch.'''
    import random
    import numpy as np
    import torch
    random.seed(n)
    np.random.seed(n)
    torch.manual_seed(n)
    torch.cuda.manual_seed_all(n)


def main(**kwargs):
    kwargs.setdefault('data_size', 500)
    kwargs.setdefault('folds', 5)
    kwargs.setdefault('epochs', 600)
    kwargs.setdefault('learning_rate', 0.001)
    kwargs.setdefault('patience', None)
    kwargs.setdefault('batch_size', 128)
    kwargs.setdefault('cuda', None)
    kwargs.setdefault('dry_run', False)
    kwargs.setdefault('name', None)
    kwargs.setdefault('seed', 1337)
    kwargs.setdefault('verbose', 'WARN')
    kwargs.setdefault('task', 'alex:nuclei')
    args = SimpleNamespace(**kwargs)

    logging.basicConfig(
        level=args.verbose,
        style='{',
        format='[{levelname:.4}][{asctime}][{name}:{lineno}] {msg}',
    )

    seed(args.seed)

    datasets = {
        'nuclei': NucleiSegmentation(n=args.data_size, k=args.folds, size=32),
        'epi': EpitheliumSegmentation(n=args.data_size, k=args.folds, size=32),
        'tubule': TubuleSegmentation(n=args.data_size, k=args.folds, size=32),
    }

    networks = {
        'alex': AlexNet(2, shape=(3, 32, 32)),
        'vgg': Vgg16(2, shape=(3, 32, 32)),
    }

    metrics = {
        'precision': precision,
        'recall': recall,
        'f-score': f_score,
    }

    if args.name is None:
        now = np.datetime64('now')
        args.name = f'{args.task}-{now}'
        logger.info(f'experiment name not given, defaulting to {args.name}')

    net, data = args.task.split(':')
    data = datasets[data]
    net = networks[net]

    # In some cases, we must move the network to it's cuda device before
    # constructing the optimizer. This is annoying, and this logic is
    # duplicated in the estimator class. Ideally, I'd like the estimator to
    # handle cuda allocation _after_ the optimizer is constructed...
    if args.cuda is None:
        args.cuda = 0 if torch.cuda.is_available() else False
    if args.cuda is not False:
        net = net.cuda(args.cuda)

    for f in range(args.folds):
        print(f'================================ Fold {f} ================================')
        train, validation, test = data.load(f)
        opt = O.Adagrad(net.parameters(), lr=args.learning_rate)
        loss = N.CrossEntropyLoss()
        model = Classifier(net, opt, loss, name=args.name, cuda=args.cuda, dry_run=args.dry_run)

        print(f'-------- Training {args.task} --------')
        model.fit(train, validation, epochs=args.epochs, patience=args.patience, batch_size=args.batch_size)
        print()

        print(f'-------- Scoring {args.task} --------')
        for metric, criteria in metrics.items():
            print(f'{metric:.10}: ', end='', flush=True)
            z = model.test(test, criteria, batch_size=args.batch_size)
            print(z)
        print()

        if args.dry_run:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run an experiment.',
        add_help=False,
        argument_default=argparse.SUPPRESS,
    )

    parser.add_argument('task', metavar='TASK', help='The task for this experiment.')

    group = parser.add_argument_group('Hyper-parameters')
    group.add_argument('-n', '--data-size', metavar='N', type=int, help='The number of training samples is a function of N.')
    group.add_argument('-k', '--folds', metavar='N', type=int, help='The number of cross-validation folds.')
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
    group.add_argument('--seed', help='Sets the random seed for the experiment, defaults to 1337.')
    group.add_argument('--name', type=str, help='Sets a name for the experiment.')
    group.add_argument('--help', action='help', help='Show this help message and exit.')

    args = parser.parse_args()
    main(**vars(args))
