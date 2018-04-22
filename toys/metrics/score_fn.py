from typing import Callable, Tuple, Union

import toys
from toys.datasets.utils import Dataset, DataLoader

from .core import Accumulator
from .supervised import NegMeanSquaredError


ScoreFn = Callable[..., Union[float, Tuple[float, ...]]]
'''A function for scoring a model's goodness of fit.

The function accepts a model to score and any number of datasets to score
against. It is unspecified how each dataset should be used. Supervised metrics
typically reserve the last dataset as the target and use the remaining as
inputs to the model. Unsupervised metrics typically use all datasets as inputs.

Arguments:
    model (Model):
        The model being scored.
    *datasets (Dataset):
        The datasets being scored against.

Returns:
    score (float or Tuple[float...]):
        The score of the model.
'''

def supervised_score(metric, **kwargs):
    '''Create a score function for supervised tasks.

    The resulting function accepts a model and at least two datasets. The final
    dataset is taken to be the targets and all others are taken to be inputs to
    the model.

    The metric must be an accumulator that accepts two observations, the target
    and the prediction, and reduces to a float. A sequence of metrics may be
    given. If so, the score function returns a tuple of floats, one for each
    metric.

    Arguments:
        metric (Accumulator or Sequence[Accumulator] or None):
            The metric(s) to be computed by the score function. Metrics must
            accept two observations, the target and the prediction, and reduce
            to a float. The default metric is negative mean squared error.
        **kwargs:
            Forwarded to the `DataLoader` constructor for each invocation of
            the score function.

    Returns:
        score (ScoreFn):
            A function for computing the given metric(s) against a
            model's predictions.
    '''
    if metric is None:
        metric = (NegMeanSquaredError(),)
    elif isinstance(metric, Accumulator):
        metric = (metric,)

    def score(model, *datasets):
        assert 1 < len(datasets)
        loader = DataLoader(*datasets, **kwargs)
        for *inputs, target in loader:
            prediction = model(*inputs)
            for m in metric:
                m.accumulate(target, prediction)
        return tuple(m.reduce() for m in metric)

    return score


def unsupervised_score(metric, **kwargs):
    '''Create a score function for unsupervised tasks.

    The resulting function accepts a model and at least one datasets. All
    datasets are taken to be inputs to the model.

    The metric must be an accumulator that accepts one observation, the
    prediction, and reduces to a float. A sequence of metrics may be given. If
    so, the score function returns a tuple of floats, one for each metric.

    Arguments:
        metric (Accumulator or Sequence[Accumulator] or None):
            The metric(s) to be computed by the score function. Metrics must
            accept one observation, the prediction, and reduce to a float.
        **kwargs:
            Forwarded to the `DataLoader` constructor for each invocation of
            the score function.

    Returns:
        score (ScoreFn):
            A function for computing the given metric(s) against a
            model's predictions.

    Todo:
        - Pick a default metric.
    '''
    if metric is None:
        raise NotImplementedError('TODO: Pick a default metric for unsupervised_score')
    elif isinstance(metric, Accumulator):
        metric = (metric,)

    def score(model, *datasets):
        loader = DataLoader(*datasets, **kwargs)
        for inputs in loader:
            prediction = model(*inputs)
            for m in metric:
                m.accumulate(prediction)
        return tuple(m.reduce() for m in metric)

    return score
