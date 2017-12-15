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
from networks import AlexNet, Vgg11, Vgg16
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
    kwargs.setdefault('data_size', 1000)
    kwargs.setdefault('patch_size', 256)
    kwargs.setdefault('ratio', '1:1:0.3')
    kwargs.setdefault('folds', 5)
    kwargs.setdefault('epochs', 600)
    kwargs.setdefault('learning_rate', 0.001)
    kwargs.setdefault('patience', None)
    kwargs.setdefault('batch_size', 48)
    kwargs.setdefault('dry_run', False)
    kwargs.setdefault('name', None)
    kwargs.setdefault('seed', 1337)
    kwargs.setdefault('verbose', 'WARN')
    kwargs.setdefault('tasks', ['alex:nuclei'])
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

    data_args = {
        'n': args.data_size,
        'k': args.folds,
        'size': args.patch_size,
        'bg_ratio': 0.1,
        'pos_ratio':  float(args.ratio.split(':')[0]),
        'edge_ratio': float(args.ratio.split(':')[1]),
        'bg_ratio':   float(args.ratio.split(':')[2]),
    }

    datasets = {
        'nuclei': NucleiSegmentation(**data_args),
        'epi': EpitheliumSegmentation(**data_args),
        'tubule': TubuleSegmentation(**data_args),
    }

    networks = {
        'alex': AlexNet((3, args.patch_size, args.patch_size), ndim=2),
        'vgg11': Vgg11((3, args.patch_size, args.patch_size), ndim=2),
        'vgg16': Vgg16((3, args.patch_size, args.patch_size), ndim=2),
    }

    if args.name is None:
        now = np.datetime64('now')
        args.name = f'exp-{now}'
        logger.info(f'experiment name not given, defaulting to {args.name}')

    for f in range(args.folds):
        print(f'================================ Fold {f} ================================')

        for task in args.tasks:
            net, data = task.split(':')

            net = networks[net].reset()
            opt = O.Adam(net.parameters())
            loss = N.CrossEntropyLoss()
            model = Classifier(net, opt, loss, name=args.name, dry_run=args.dry_run)

            data = datasets[data]
            train, validation, test = data.load(f)

            print(f'-------- Fitting {task} --------')
            reports = {'f-score': FScore()}
            model.fit(train, validation, epochs=args.epochs, patience=args.patience, reports=reports, batch_size=args.batch_size)
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

            # move net back to the cpu before starting the next task
            net.cpu()

        if args.dry_run:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        add_help=False,
        argument_default=argparse.SUPPRESS,
        formatter_class=argparse.RawTextHelpFormatter,
        description=(
            'Runs an experiment.\n'
            '\n'
            'Tasks are given as NETWORK:DATASET where NETWORK names a supported network\n'
            'architecture and DATASET names one of the pathology datasets.\n'
            '\n'
            'Note that the experiment is intended to be executed from the root of the\n'
            'repository using `python -m`:\n'
            '\n'
            '  python -m experiments.pathology alex:nuclei\n'
            '\n'
            'Networks:\n'
            '  alex     A reduced AlexNet, designed for CIFAR-10\n'
            '  vgg11    VGG11 a.k.a. VGG-A\n'
            '  vgg16    VGG16 a.k.a. VGG-D\n'
            '\n'
            'Datasets:\n'
            '  nuclei   A nuclei segmentation dataset\n'
            '  epi      An epithelium segmentation dataset\n'
        ),
    )

    group = parser.add_argument_group('Hyper-parameters')
    group.add_argument('-n', '--data-size', metavar='X', type=int)
    group.add_argument('-p', '--patch-size', metavar='X', type=int)
    group.add_argument('-r', '--ratio', metavar='X', type=int)
    group.add_argument('-k', '--folds', metavar='X', type=int)
    group.add_argument('-e', '--epochs', metavar='X', type=int)
    group.add_argument('-l', '--learning-rate', metavar='X', type=float)
    group.add_argument('-z', '--patience', metavar='X', type=int)

    group = parser.add_argument_group('Performance')
    group.add_argument('-b', '--batch-size', metavar='X', type=int)

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
