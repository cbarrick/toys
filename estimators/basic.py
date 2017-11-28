import sys

import torch
import torch.autograd as A
import torch.nn.functional as F
import torch.utils.data as D


class Estimator:
    '''Wraps a torch network, optimizer, and loss to an sklearn-like estimator.
    '''

    def __init__(self, net, opt, loss, cuda=None, dry_run=False):
        '''Create a basic estimator.

        Args:
            net: The network to train.
            opt: The optimizer to step during training.
            loss: The loss function to minimize.
            cuda: The cuda device. `None` means automatic. `False` means disabled.
            dry_run: Cut loops short, for testing.
        '''
        if cuda is None and torch.cuda.is_available():
            cuda = 0
            net = net.cuda(cuda)
        elif cuda is None:
            cuda = False

        self.net = net
        self.opt = opt
        self.loss = loss
        self.cuda = cuda
        self.dry_run = dry_run

    def reset(self):
        '''Reset the estimator to it's initial state.
        '''
        self.net.reset()
        return self

    def variable(self, x, **kwargs):
        '''Cast a tensor to a variable.

        Args:
            x: The tensor to wrap.
            **kwargs: Passed to the `Variable` constructor.

        Returns:
            Returns a torch `Variable` on the same cuda device as the network.
            If x is already a `Variable`, it is not wrapped, but it is moved to the GPU.
        '''
        if not isinstance(x, A.Variable):
            x = A.Variable(x, **kwargs)
        if self.cuda is not False:
            x = x.cuda(self.cuda, async=True)
        return x

    def params(self):
        '''Get the list of trainable paramaters.

        Returns:
            Returns a list of all parameters known to the optimizer.
        '''
        return [p for group in self.opt.param_groups for p in group['params']]

    def partial_fit(self, x, y):
        '''Performs one step of the optimization.

        Args:
            x: The input batch.
            y: The class labels.

        Returns:
            Returns the losses for this batch.
        '''
        self.net.train()
        self.opt.zero_grad()
        x = self.variable(x)
        y = self.variable(y)
        h = self.net(x)
        j = self.loss(h, y)
        j.backward()
        self.opt.step()
        return j.data

    def fit(self, data, validation, max_epochs=100, patience=5, **kwargs):
        '''Fit the model to a dataset.

        Args:
            data: A dataset to fit.
            validation: A dataset to use as the validation set.
            max_epochs: The maximum number of epochs to spend training.
            patience: Stop if the validation loss does not improve after this many epochs.
            **kwargs: Forwarded to torch's `DataLoader` class, with more sensible defaults.

        Returns:
            Returns the validation loss.
            Returns train loss if no validation set is given.
        '''
        kwargs.setdefault('shuffle', True)
        kwargs.setdefault('pin_memory', self.cuda is not None)
        data = D.DataLoader(data, **kwargs)

        best_loss = float('inf')
        p = patience
        for epoch in range(max_epochs):

            # Train
            train_loss = 0
            print(f'epoch {epoch+1} [0%]', end='\r', flush=True, file=sys.stderr)
            for i, (x, y) in enumerate(data):
                j = self.partial_fit(x, y)
                train_loss += j.sum() / len(data.dataset)
                progress = (i+1) / len(data)
                print(f'epoch {epoch+1} [{progress:.2%}]', end='\r', flush=True, file=sys.stderr)
                if self.dry_run:
                    break
            print(f'epoch {epoch+1} [Train loss: {train_loss:8.6f}]', end='')

            # Validate
            val_loss = self.test(validation)
            print(f' [Validation loss: {val_loss:8.6f}]', flush=True)

            # Convergence test
            if val_loss < best_loss:
                best_loss = val_loss
                p = patience
            else:
                p -= 1
                if p == 0:
                    break

        return val_loss

    def predict(self, x):
        '''Apply the network to some input batch.

        Args:
            x: The input batch.

        Returns:
            Returns the output of the network.
        '''
        self.net.eval()
        x = self.variable(x, volatile=True)
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
            x = self.variable(x, volatile=True)
            y = self.variable(y, volatile=True)
            h = self.net(x)
            j = self.loss(h, y)
            return j.data.mean()

        h = self.predict(x)
        try:
            return criteria(y, h)
        except (ValueError, TypeError):
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
        kwargs.setdefault('drop_last', True)
        kwargs.setdefault('pin_memory', self.cuda is not None)
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
