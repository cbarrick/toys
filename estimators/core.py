import logging
import sys
from pathlib import Path

import torch
import torch.autograd as A
import torch.nn.functional as F
import torch.utils.data as D


logger = logging.getLogger(__name__)


class Estimator:
    '''Wraps a torch network, optimizer, and loss to an sklearn-like estimator.
    '''

    def __init__(self, net, opt, loss, name='model', cuda=None, dry_run=False):
        '''Create basic estimator.

        Args:
            net: The network to train.
            opt: The optimizer to step during training.
            loss: The loss function to minimize.
            name: A name for the model.
            cuda: The cuda device to use.
            dry_run: Cut loops short.
        '''
        if cuda is None:
            cuda = 0 if torch.cuda.is_available() else False
        if cuda is not False:
            net = net.cuda(cuda)

        self.net = net
        self.opt = opt
        self.loss = loss
        self.name = name
        self.cuda = cuda
        self.dry_run = dry_run
        self.reset()

        self.path = Path(f'./checkpoints/{self.name}.torch')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch()

    def reset(self):
        '''Reset the estimator to it's initial state.
        '''
        self.net.reset()
        return self

    def save(self, path=None):
        '''Saves the model parameters to disk.
        '''
        if path is None:
            path = self.path
        state = self.net.state_dict()
        torch.save(state, str(path))
        return self

    def load(self, path=None):
        '''Loads the model parameters from disk.
        '''
        if path is None:
            path = self.path
        state = torch.load(str(path))
        self.net.load_state_dict(state)
        return self

    def variable(self, x, **kwargs):
        '''Cast a tensor to a `Variable` on the same cuda device as the network.

        If the input is already a `Variable`, it is not wrapped.

        Args:
            x: The tensor to wrap.
            **kwargs: Passed to the `Variable` constructor.
        '''
        if not isinstance(x, A.Variable):
            x = A.Variable(x, **kwargs)
        if self.cuda is not False:
            x = x.cuda(self.cuda, async=True)
        return x

    def tensor(self, x):
        '''Cast some `x` to a torch `Tensor` on the same cuda device as the network.

        Args:
            x: The input to cast
        '''
        if isinstance(x, A.Variable):
            x = x.detach().data
        if self.cuda is not False:
            x = x.cuda(self.cuda, async=True)
        return x

    def params(self):
        '''Get the list of trainable paramaters.

        Returns:
            A list of all parameters in the optimizer's `param_groups`.
        '''
        return [p for group in self.opt.param_groups for p in group['params']]

    def partial_fit(self, x, y):
        '''Performs one step of the optimization.

        Args:
            x: The input batch.
            y: The class labels.

        Returns:
            Returns the average loss for this batch.
        '''
        self.net.train()  # put the net in train mode, effects dropout layers etc.
        self.opt.zero_grad()  # reset the gradients of the trainable variables.
        x = self.variable(x)
        y = self.variable(y)
        h = self.net(x)
        j = self.loss(h, y)
        j.backward()
        self.opt.step()
        return j.data.mean()

    def fit(self, train, validation=None, epochs=100, patience=50, **kwargs):
        '''Fit the model to a dataset.

        Args:
            train: A dataset to fit.
            validation: A dataset to use as the validation set.
            epochs: The maximum number of epochs to spend training.
            patience: Stop if the validation loss does not improve after this many epochs.
            **kwargs: Forwarded to torch's `DataLoader` class, with more sensible defaults.

        Returns:
            Returns the validation loss.
            Returns train loss if no validation set is given.
        '''
        kwargs.setdefault('shuffle', True)
        kwargs.setdefault('pin_memory', self.cuda is not False)
        train = D.DataLoader(train, **kwargs)

        best_loss = float('inf')
        p = patience or -1
        for epoch in range(epochs):

            # Train
            n = len(train)
            train_loss = 0
            print(f'epoch {epoch+1} [0%]', end='\r', flush=True, file=sys.stderr)
            for i, (x, y) in enumerate(train):
                j = self.partial_fit(x, y)
                train_loss += j / n
                progress = (i+1) / n
                print(f'epoch {epoch+1} [{progress:.2%}]', end='\r', flush=True, file=sys.stderr)
                if self.dry_run:
                    break
            print('\001b[2K', end='\r', flush=True, file=sys.stderr)  # ANSI escape code to clear line
            print(f'epoch {epoch+1}', end=' ', flush=True)
            print(f'[Train loss: {train_loss:8.6f}]', end=' ', flush=True)

            # Validate
            if validation:
                val_loss = self.test(validation, **kwargs)
                print(f'[Validation loss: {val_loss:8.6f}]', end=' ', flush=True)

            # Early stopping
            loss = val_loss if validation else train_loss
            if loss < best_loss:
                best_loss = loss
                p = patience or -1
                self.save()
                print('✓', end=' ', flush=True)
            else:
                p -= 1

            print()
            if p == 0:
                break

        self.load()
        return loss

    def predict(self, x):
        '''Apply the network to some input batch.

        Args:
            x: The input batch.

        Returns:
            Returns the output of the network.
        '''
        self.net.eval()  # put the net in eval mode, effects dropout layers etc.
        x = self.variable(x, volatile=True)  # use volatile input to save memory when not training.
        h = self.net(x)
        return h.data

    def score(self, x, y, criteria=None):
        '''Score the model on a batch of inputs and labels.

        Args:
            x: The input batch.
            y: The targets.
            criteria: The metric to measure; defaults to the mean loss.

        Returns:
            Returns the result of `criteria(true, predicted)`.
        '''
        self.net.eval()

        if criteria is None:
            # Default - the loss has a different signature than a metric
            x = self.variable(x, volatile=True)
            y = self.variable(y, volatile=True)
            h = self.net(x)
            j = self.loss(h, y)
            return j.data.mean()

        try:
            # Fast path - try to use the GPU
            h = self.predict(x)
            y = self.tensor(y)
            return criteria(y, h)

        except (ValueError, TypeError):
            # Slow path - most sklearn metrics cannot handle Tensors
            logger.debug('computing a slow metric')
            h = h.cpu().numpy()
            y = y.cpu().numpy()
            return criteria(y, h)

    def test(self, data, criteria=None, **kwargs):
        '''Score the model on a dataset.

        Args:
            data: A dataset to score against.
            criteria: The metric to measure; defaults to the loss.
            **kwargs: Forwarded to torch's `DataLoader` class, with more sensible defaults.

        Returns:
            Returns the result of `criteria(true, predicted)` averaged over all batches.
            The last incomplete batch is dropped by default.
        '''
        kwargs.setdefault('pin_memory', self.cuda is not False)
        kwargs['drop_last'] = True
        data = D.DataLoader(data, **kwargs)
        n = len(data)
        loss = 0
        for x, y in data:
            j = self.score(x, y, criteria)
            loss += j / n
            if self.dry_run:
                break
        return loss


class Classifier(Estimator):
    def predict(self, x):
        '''Classify some input batch.

        Args:
            x: The input batch.

        Returns:
            Returns the class numbers of each batch item.
        '''
        self.net.eval()
        x = self.variable(x, volatile=True)
        h = self.net(x)
        _, h = h.max(1)
        return h.data

    def predict_proba(self, x):
        '''Compute the likelihoods for some input batch.

        Args:
            x: The input batch.

        Returns:
            The softmax of the network's output.
        '''
        self.net.eval()
        x = self.variable(x, volatile=True)
        h = self.net(x)
        h = F.softmax(h)
        return h.data
