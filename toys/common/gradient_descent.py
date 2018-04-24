from typing import Any, Mapping, Sequence
import logging

import torch
from torch.autograd import Variable
from torch.nn import DataParallel, Module
from torch.optim import Optimizer

import toys
from toys import Estimator, Model
from toys.metrics import Mean
from toys.datasets.utils import Dataset, DataLoader

from .torch import TorchModel, TorchDtype

logger = logging.getLogger(__name__)


class GradientDescent(Estimator):
    '''A supervised stochastic gradient descent estimator for PyTorch modules.
    '''

    def __init__(self, module, **defaults):
        '''Construct a GradientDescent estimator.

        Arguments:
            module (Module):
                A constructor for the PyTorch module to train.
            **defaults (Mapping[str, Any]):
                Overrides the default arguments to `fit`.
        '''
        super().__init__(**defaults)
        self.module = module

    def fit(self, dataset, loss_fn='mse_loss', optimizer='SGD:lr=1e-4', max_epochs=100,
            batch_size=1, device_ids=None, stop_policy=None, patience=0, dtype=None,
            **kwargs):
        '''Trains a TorchModel.

        Users should not call this method, but instead call the estimator
        directly.

        Arguments:
            dataset (Dataset):
                The dataset to fit.
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
            dtype (str or TorchDtype or None):
                Cast the module to this data type. This can be a PyTorch tensor
                class, a conventional name like 'float' and 'double', or an
                explicit name like 'float32' and 'float64'. The default is
                determined by ``torch.Tensor`` and may be overridden with
                ``torch.set_default_tensor_type``.
            **kwargs (Mapping[str, Any]):
                Additional arguments passed to the module constructor.

        Returns:
            model (TorchModel):
                A model wrapping the learned module. Note that the module is
                moved to the CPU even if it was trained using GPUs.
        '''
        proto_x, proto_y = dataset[0]
        in_features = proto_x.shape[-1]
        out_features = proto_y.shape[-1]

        if device_ids is None:
            device_count = torch.cuda.device_count()
            device_ids = list(range(device_count))

        dtype = torch_dtype(dtype)

        mod = self.module(in_features, out_features, **kwargs)
        mod = mod.type(dtype).train()
        mod = DataParallel(mod, device_ids)

        if isinstance(optimizer, str):
            optimizer = parse_optimizer(optimizer)
        opt = optimizer(mod.parameters())

        if isinstance(loss_fn, str):
            loss_fn = parse_loss(loss_fn)
        assert callable(loss_fn)

        p = patience  # early stopping counter

        if stop_policy is None:
            stop_policy = lambda _: False  # never stop

        # Helper to repeatedly print messages over each other on the same line.
        # Note that the cursor is left on the same line.
        def progress(*vals, sep=' '):
            print('\u001b[2K', end='\r')  # CSI escape code to clear the line
            print(*vals, end='', sep=sep, flush=True)

        # Construct the DataLoader.
        def dataloader(**kwargs):
            kwargs.setdefault('batch_size', batch_size)
            return DataLoader(dataset, **kwargs)

        # Perform one iteration of gradient descent.
        def partial_fit(x, y):
            opt.zero_grad()
            if device_ids: x = x.cuda(async=True)
            if device_ids: y = y.cuda(async=True)
            x = Variable(x).type(dtype)
            y = Variable(y).type(dtype)
            h = mod(x)
            j = loss_fn(h, y)
            j.backward()
            opt.step()
            return j.data

        # Perform one epoch of gradient descent.
        def train_epoch():
            train_set = dataloader(shuffle=True)
            train_loss = Mean()
            n = len(train_set)
            for i, batch in enumerate(train_set):
                progress(f'[{i/n:.2%}]')
                j = partial_fit(*batch)
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

        mod = mod.module  # Unwrap out of DataParallel.
        mod = mod.eval()  # Exit of training mode.
        mod = mod.cpu()   # Release GPU resources.
        return TorchModel(mod, dtype)
