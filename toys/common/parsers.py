import torch
from torch import nn, optim


def parse_str(s):
    kwargs = {}
    head, *parts = s.split(':')
    for i, part in enumerate(parts):
        param, val = part.split('=', 1)
        try:
            val = float(val)
        except ValueError:
            pass
        kwargs[param] = val
    return head, kwargs


def parse_activation(act):
    act, kwargs = parse_str(act)

    if act == 'sigmoid': return nn.Sigmoid(**kwargs)
    if act == 'tanh': return nn.Tanh(**kwargs)
    if act == 'relu': return nn.ReLU(**kwargs, inplace=True)
    if act == 'relu6': return nn.ReLU6(**kwargs, inplace=True)
    if act == 'elu': return nn.ELU(**kwargs, inplace=True)
    if act == 'selu': return nn.SELU(**kwargs, inplace=True)
    if act == 'prelu': return nn.PReLU(**kwargs)
    if act == 'leaky_relu': return nn.LeakyReLU(**kwargs, inplace=True)
    if act == 'threshold': return nn.Threshold(**kwargs, inplace=True)
    if act == 'hardtanh': return nn.Hardtanh(**kwargs, inplace=True)
    if act == 'log_sigmoid': return nn.LogSigmoid(**kwargs)
    if act == 'softplus': return nn.Softplus(**kwargs)
    if act == 'softshrink': return nn.Softshrink(**kwargs)
    if act == 'tanhshrink': return nn.Tanhshrink(**kwargs)
    if act == 'softmin': return nn.Softmin(**kwargs)
    if act == 'softmax': return nn.Softmax(**kwargs)
    if act == 'softmax2d': return nn.Softmax2d(**kwargs)
    if act == 'log_softmax': return nn.LogSoftmax(**kwargs)

    raise ValueError(f'unknown activation: {repr(act)}')


def parse_initializer(init):
    init, kwargs = parse_str(init)

    if init == 'uniform': return lambda x: nn.init.uniform(x, **kwargs)
    if init == 'normal': return lambda x: nn.init.normal(x, **kwargs)
    if init == 'constant': return lambda x: nn.init.constant(x, **kwargs)
    if init == 'eye': return lambda x: nn.init.eye(x, **kwargs)
    if init == 'dirac': return lambda x: nn.init.dirac(x, **kwargs)
    if init == 'xavier': return lambda x: nn.init.xavier_uniform(x, **kwargs)
    if init == 'xavier_uniform': return lambda x: nn.init.xavier_uniform(x, **kwargs)
    if init == 'xavier_normal': return lambda x: nn.init.xavier_normal(x, **kwargs)
    if init == 'kaiming': return lambda x: nn.init.kaiming_uniform(x, **kwargs)
    if init == 'kaiming_uniform': return lambda x: nn.init.kaiming_uniform(x, **kwargs)
    if init == 'kaiming_normal': return lambda x: nn.init.kaiming_normal(x, **kwargs)
    if init == 'orthogonal': return lambda x: nn.init.orthogonal(x, **kwargs)
    if init == 'sparse': return lambda x: nn.init.sparse(x, **kwargs)

    # `init` may also be the name of an activation function,
    # in which case we return `xavier_uniform` with the proper gain.
    # If `init` is unknown, we use a gain of 1.
    if init == 'leaky_relu':
        param = kwargs.get('negative_slope', 0.01)
    else:
        param = None
    gain = nn.init.calculate_gain(init, param)
    gain = kwargs.get('gain', gain)
    return lambda x: nn.init.xavier_uniform(x, gain)


def parse_optimizer(opt):
    opt, kwargs = parse_str(opt)
    opt = getattr(optim, opt)
    return lambda params: opt(params, **kwargs)


def parse_loss(loss):
    loss, kwargs = parse_str(loss)

    if loss == 'l1': return nn.L1Loss(**kwargs)
    if loss == 'mse': return nn.MSELoss(**kwargs)
    if loss == 'cross_entropy': return nn.CrossEntropyLoss(**kwargs)
    if loss == 'nll': return nn.NLLLoss(**kwargs)
    if loss == 'poisson': return nn.PoissonLoss(**kwargs)
    if loss == 'nll2d': return nn.NLLLoss2d(**kwargs)
    if loss == 'kl_div': return nn.KLDivLoss(**kwargs)
    if loss == 'bce': return nn.BCELoss(**kwargs)
    if loss == 'bce_with_logits': return nn.BCEWithLogitsLoss(**kwargs)
    if loss == 'margin_ranking': return nn.MarginRankingLoss(**kwargs)
    if loss == 'hinge_embedding': return nn.HingeEmbeddingLoss(**kwargs)
    if loss == 'multilabel_margin': return nn.MultiLabelMarginLoss(**kwargs)
    if loss == 'smooth_l1': return nn.SmoothL1Loss(**kwargs)
    if loss == 'multilabel_softmargin': return nn.MultiLabelSoftMarginLoss(**kwargs)
    if loss == 'cosine_embedding': return nn.CosineEmbeddingLoss(**kwargs)
    if loss == 'multi_margin': return nn.MultiMarginLoss(**kwargs)
    if loss == 'triplet_margin': return nn.TripletMarginLoss(**kwargs)

    loss = getattr(nn.functional, loss)
    return lambda x: loss(x, **kwargs)
