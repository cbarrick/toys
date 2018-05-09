from typing import Callable, Sequence

import torch
from torch.nn import DataParallel, Module
from torch.optim import Optimizer

import toys
from toys.common import BaseEstimator, TorchModel
from toys.datasets.utils import Dataset, DataLoader
from toys.metrics import Mean
from toys.parsers import parse_dtype, parse_loss, parse_optimizer


class GradientDescent(BaseEstimator):
    '''A supervised stochastic gradient descent estimator for PyTorch modules.
    '''
    def fit(self, dataset, *, module=None, loss_fn='mse', optimizer='SGD:lr=1e-4', max_epochs=100,
            batch_size=1, device_ids=None, stop_policy=None, patience=-1, dtype=None, **kwargs):
        '''Trains a TorchModel.

        Users should not call this method directly, but instead call the
        estimator as a function.

        Arguments:
            dataset (Dataset):
                The data to fit. The elements of the dataset should be
                $n$-tuples. In supervised mode, the first $n-1$ elements are
                used as inputs to the module, with the final element as the
                target. In unspuervised mode, all elements are input to the
                module.

        Keyword Arguments:
            module (Module or None):
                A constructor for the PyTorch module to train. The module may
                be specified either when constructing or calling this estimator
                and MUST NOT be None. (Though None will successfully typecheck.)
            loss_fn (str or Callable[..., float]):
                The loss function. If a string is passed, it is looked up in
                the `torch.nn.functional` module. Otherwise, the argument must
                be a function which takes the predicted values and the true
                targets and returns the computed loss.
            optimizer (str or Callable[..., Optimizer]):
                A constructor for the optimizer. If a string is given, it is
                parsed as the name of a class in the `torch.optim` module,
                optionally followed by keyword arguments of the form
                ':{KEY}={VAL}' (note the leading ':'). Values will be cast to
                float when possible. If a callable is given, it should take the
                trainable parameters and return an optimizer.
            max_epochs (int):
                The maximum number of passes over the data during training.
            batch_size (int):
                The batch size for each iteration of the optimizer. To maximize
                GPU utilization, it should be an integer multiple of the number
                of devices.
            device_ids (Sequence[int] or None):
                A list of CUDA device IDs to use during training.
                The default is to use all devices.
            stop_policy (Callable[[float], bool] or None):
                Determines when to halt learning. The argument must be a
                function which accepts the mean validation or training loss at
                the end of each epoch and returns true if training should halt.
                The default is to never stop early.
            patience (int):
                The stop policy must return true this many additional times
                consecutivly to stop training. A negative value is equivalent
                to an infinite patience.
            dtype (str or torch.dtype or None):
                Cast the module to this data type. This can be a PyTorch tensor
                class, a conventional name like 'float' and 'double', or an
                explicit name like 'float32' and 'float64'. The default is
                determined by `torch.get_default_dtype` and may be set with
                `torch.set_default_dtype`.
            **kwargs:
                Additional keyword arguments are forwarded to the module
                constructor.

        Returns:
            model (TorchModel):
                A model wrapping the learned module. Note that the module is
                moved to the CPU even if it was trained using GPUs.
        '''
        device_count = torch.cuda.device_count()
        all_devices = list(range(device_count))
        never_stop = lambda _: False

        device_ids = device_ids or all_devices
        stop_policy = stop_policy or never_stop
        optimizer = parse_optimizer(optimizer)
        loss_fn = parse_loss(loss_fn)
        dtype = parse_dtype(dtype)

        mod = module(**kwargs)
        mod = mod.to(dtype).train()
        mod = DataParallel(mod, device_ids)
        opt = optimizer(mod.parameters())

        # Helper to repeatedly print messages over each other on the same line.
        # Note that the cursor is left on the same line.
        def progress(*vals, sep=' '):
            print('\u001b[2K', end='\r')  # CSI escape code to clear the line
            print(*vals, end='', sep=sep, flush=True)

        # Perform one iteration of gradient descent.
        def partial_fit(batch):
            opt.zero_grad()
            if unsupervised:
                prediction = mod(*batch)
                loss = loss_fn(prediction)
            else:
                *inputs, target = batch
                prediction = mod(*inputs)
                loss = loss_fn(prediction, target)
            loss.backward()
            opt.step()
            return loss.detach()

        # Perform one epoch of gradient descent.
        def train_epoch():
            train_set = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            train_loss = Mean()
            n = len(train_set)
            for i, batch in enumerate(train_set):
                progress(f'[{i/n:.2%}]')
                j = partial_fit(batch)
                train_loss.accumulate(j)
            return train_loss.reduce()

        # Compute the loss of the validation set, if given.
        def validate():
            pass

        # Print a report at the end of an epoch.
        def report(epoch, val_loss):
            print('\u001b[2K', end='\r')  # CSI escape code to clear the line
            print(f'[epoch {epoch+1}]', end='\t')
            print(f'[loss: {val_loss:0.4e}]', end='\t')
            print()
            pass

        # Takes the validation loss from each epoch to determine when to stop.
        # This just wraps the `stop_policy` with a patience counter.
        p = patience
        def stop(val_loss):
            nonlocal p
            if stop_policy(val_loss):
                p -= 1
                return p == -1
            else:
                p = patience
                return False

        # The actual training loop.
        for epoch in range(max_epochs):
            train_loss = train_epoch()
            val_loss = validate() or train_loss
            report(epoch, val_loss)
            if stop(val_loss): break

        mod = mod.module  # Unwrap from DataParallel.
        mod = mod.eval()  # Exit training mode.
        mod = mod.cpu()   # Release GPU resources.

        proto = x[0]
        shape = proto.shape
        dim = len(shape)

        return TorchModel(mod, dim)