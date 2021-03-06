from typing import Callable, Sequence

import torch
from torch.nn import DataParallel, Module
from torch.optim import Optimizer

import toys
from toys.common import BaseEstimator, TorchModel
from toys.data import Dataset
from toys.metrics import Mean
from toys.parsers import parse_dtype, parse_loss, parse_optimizer


class GradientDescent(BaseEstimator):
    '''A supervised stochastic gradient descent estimator for PyTorch modules.
    '''
    def __init__(self, ctor=None, **kwargs):
        kwargs['ctor'] = ctor
        super().__init__(**kwargs)

    def fit(self, *datasets, ctor=None, loss_fn='cross_entropy', optimizer='SGD:lr=1e-4',
            max_epochs=100, batch_size=1, device_ids=None, stop_policy=None, patience=-1,
            dtype=None, dry_run=False, **kwargs):
        '''Trains a TorchModel.

        Users should not call this method directly, but instead call the
        estimator as a function.

        Arguments:
            datasets (Dataset):
                The datasets to fit. If more than one are given, they are
                combined using `toys.zip`. The target is taken from the last
                column.

        Keyword Arguments:
            ctor (Module or None):
                A constructor for the PyTorch module to train. The ctor may
                be specified either when constructing or calling this estimator
                and MUST NOT be None. (Though None will successfully typecheck.)
            loss_fn (str or Callable[..., float]):
                The loss function. If the argument is a function, it must
                accept the predicted values and the true targets as arguments
                and return the computed loss.
            optimizer (str or Callable[..., Optimizer]):
                A constructor for the optimizer. If the argument is a function,
                it should take the trainable parameters as an argument and
                return an optimizer.
            max_epochs (int):
                The maximum number of passes over the data during training.
                The default is 100.
            device_ids (Sequence[int]):
                A list of CUDA device IDs to use during training.
                The default is to use all devices.
            stop_policy (Callable[[float], bool]):
                Determines when to halt learning. The argument must be a
                function which accepts the mean validation or training loss at
                the end of each epoch and returns true if training should halt.
                The default is to never stop early.
            patience (int):
                The stop policy must return true this many additional times
                consecutivly to stop training. The default is -1, meaning
                infinite patience.
            dtype (str or torch.dtype):
                Cast the module to this data type. This can be a PyTorch dtype
                object, a conventional name like 'float' and 'double', or an
                explicit name like 'float32' and 'float64'. The default is
                determined by `torch.get_default_dtype` and may be set with
                `torch.set_default_dtype`.
            dry_run (bool):
                If true, break from loops early. Useful for debugging.
            **kwargs:
                Additional keyword arguments are forwarded to the module
                constructor. Common arguments include ``in_shape`` and
                ``out_shape``.

        Returns:
            model (TorchModel):
                A model wrapping the learned module. Note that the module is
                moved to the CPU even if it was trained using GPUs.
        '''
        dataset = toys.zip(*datasets)

        device_count = torch.cuda.device_count()
        use_cuda = bool(device_count)
        all_devices = list(range(device_count))
        device_ids = device_ids or all_devices

        never_stop = lambda _: False
        stop_policy = stop_policy or never_stop

        dtype = dtype or torch.get_default_dtype()
        dtype = parse_dtype(dtype)

        loss_fn = parse_loss(loss_fn)

        assert ctor is not None
        mod = TorchModel(ctor(**kwargs), device_ids, dtype)

        optimizer = parse_optimizer(optimizer)
        opt = optimizer(mod.parameters())

        # Helper to repeatedly print messages over each other on the same line.
        # Note that the cursor is left on the same line.
        def progress(*vals, sep=' '):
            print('\u001b[2K', end='\r')  # CSI escape code to clear the line
            print(*vals, end='', sep=sep, flush=True)

        # Perform one iteration of gradient descent.
        def partial_fit(batch):
            opt.zero_grad()
            *features, target = batch
            if use_cuda: target = target.cuda()
            prediction = mod(*features)
            loss = loss_fn(prediction, target)
            loss = loss.mean()
            loss.backward()
            opt.step()
            return loss.detach()

        # Perform one epoch of gradient descent.
        def train_epoch():
            train_set = toys.batches(dataset, batch_size)
            train_loss = Mean()
            n = len(train_set)
            for i, batch in enumerate(train_set):
                progress(f'[{i/n:.2%}]')
                j = partial_fit(batch)
                train_loss.accumulate([j])
                if dry_run: break
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
            if dry_run: break

        mod = mod.eval()
        return mod
