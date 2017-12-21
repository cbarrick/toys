import logging

import numpy as np
import torch

import autograd as A
import datasets as D
import estimators as E
import metrics as M
import networks.functional as F


logger = logging.getLogger(__name__)


class EwcClassifier(E.Classifier):
    '''An classifier that learns multiple tasks through elastic weight consolidation.

    Elastic weight consolidation (EWC) is a method for training a single model
    to perform multiple tasks. The idea is that we first train the model to
    perform a single task, then we "consolidate" the parameters into a new
    regularization term on the loss function before we start training on a new
    task. This regularization allows the model to learn the new task while
    keeping parameters which are important for the first task near to their
    original values. Once we have trained on the second task, we can then
    consolidate the weights again before training on a new task.

    References:
        "Overcoming catastrophic forgetting in neural networks" https://arxiv.org/abs/1612.00796
    '''

    def reset(self):
        super().reset()
        self.ewc = []

    def fisher_information(self, x, y):
        '''Estimate the Fisher information of the trainable parameters.

        Args:
            x: Some input samples.
            y: Some class labels.

        Returns:
            Returns the fisher information of the trainable parameters.
            The values are arranged similarly to `EwcClassifier.params()`.
        '''
        # Fisher information is the variance of the gradient of the log-likelihood,
        # i.e. E[(𝛻 log f(X;𝜃)) ** 2 | 𝜃]
        # This is a relativly straightforward implementation of that formula
        # for classification. Note that we must backprop with a scalar, so we
        # commute the summation part of the expectation across the derivative.
        # BUT using a sum makes this numerically unstable. I have 2 questions:
        # 1. How can we make this more stable? Can we commute the entire mean
        #    across the derivative? I think not...
        # 2. How do we generalize this to continuous regression problems?
        #    i.e. How should we interpret log-likelihood in that case?
        self.net.eval()
        self.opt.zero_grad()
        x = self.variable(x)
        y = self.variable(y)
        h = self.net(x)
        l = F.log_softmax(h)[range(y.size(0)), y.data]  # log-likelihood of true class
        l = l.sum()
        l.backward()
        grads = (p.grad.data.clone() for p in self.params())
        fisher = [(g ** 2) / len(x) for g in grads]
        return fisher

    def consolidate(self, data, alpha=1, **kwargs):
        '''Extend the loss function with an EWC regularization term.

        This method adds an L2-regularization term that constrains the
        parameters to their current value. The strength of this regularization
        is stronger for parameters which are more important for the current
        task, as determined by their Fisher information. The Fisher information
        is estimated from a representative sample of the current task.

        Args:
            data: Samples of the current task, typically the validation set.
            alpha: A global regularization strength for the new term.

        Kwargs:
            Forwarded to torch's `DataLoader` class, except:
            pin_memory: Defaults to True if the estimator is using cuda.
        '''
        kwargs.setdefault('pin_memory', self.cuda is not False)
        data = D.DataLoader(data, **kwargs)

        params = [p.clone() for p in self.params()]
        fisher = [M.Mean() for p in self.params()]

        for x, y in data:
            for i, f in enumerate(self.fisher_information(x, y)):
                fisher[i].accumulate(f)
            if self.dry_run: break
        fisher = [f.reduce() for f in fisher]

        self.ewc += [{
            'params': params,
            'fisher': fisher,
            'alpha': alpha,  # The name 'lambda' is taken by the keyword.
        }]

    def loss(self, *args, **kwargs):
        '''Compute the loss with EWC regularization.
        '''
        j = self._loss(*args, **kwargs)

        # Return the normal loss if there are no consolidated tasks.
        if len(self.ewc) == 0:
            return j

        # Add the regularization for each consolidated task.
        params = self.params()
        for term in self.ewc:
            a = term['alpha']
            ewc = ((p - t) ** 2 for t, p in zip(term['params'], params))
            ewc = (f * e for f, e in zip(term['fisher'], ewc))
            ewc = (a/2 * e for e in ewc)
            ewc = (e.sum() for e in ewc)
            j += sum(ewc)

        return j
