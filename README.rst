================================================================================
               Toys: Friendly machine learning for rapid research
================================================================================

Toys is a framework for machine learning experiments inspired by Scikit-learn and built with PyTorch.


Development Roadmap
===================
The implementation of Toys is just getting started. My initial effort is being put towards implementing training algorithms which wrap PyTorch modules, implementing model selection algorithms like ``GridSearchCV``, and implementing the most important scoring metrics. In the long run, I'd like to support Dask arrays for compatibility with that ecosystem of out-of-core datasets and distributed computation, but currently there are upstream compatibility issues between Dask and PyTorch. The hope is to one day reach feature parity with Scikit-learn, but this will certainly require a healthy community of contributors. To reach that point, I first need to develop the infrastructure and design guidelines to foster distributed development.

Listed below are tentative project goals roughly organized by priority.

Highest Priority
----------------
- Core API design
- Infrastructure (setup.py, Sphinx, documentation, hosting)
- Generic PyTorch wrappers
- Parameter tuning, grid search
- Common metrics, model validation
- Serialization

High Priority
-------------
- Higher level neural net layers (see `tf.layers`_ and `tf.variable_scope`_)
- Common neural net architectures (ResNet, U-Net)
- Linear classifiers and regressors (linear, softmax, SVM)
- Decision trees and ensembles (CART, random forest, boosted trees)
- Naive Bayes
- Preprocessing (standard scaling, TF-IDF)
- Common datasets (MNIST, CIFAR)

Medium Priority
---------------
- `Dask`_ support
- Bayesian inference, probabilistic graphical models (see `Pyro`_)
- Reinforcement learning (see `Gym`_)
- Unsupervised learning (K-NN, K-Means, spectral clustering)
- Matrix decomposition / feature extraction (PCA, NMF, etc)
- Feature selection

Low Priority
------------
- A tool to extract MyPy type annotation stubs from docstrings
- Scikit-learn feature parity


.. _Dask: https://dask.pydata.org/en/latest/
.. _Gym: https://gym.openai.com/
.. _Pyro: http://pyro.ai/
.. _PyTorch: http://pytorch.org/
.. _tf.layers: https://www.tensorflow.org/api_guides/python/contrib.layers
.. _tf.variable_scope: https://www.tensorflow.org/api_docs/python/tf/variable_scope


Contributing
============

This is project is a large endeavor and all are welcome to contribute. But because the project is so young, coordination is key. Please reach out on the issue tracker, or in person if you are around UGA, if you are interested in contributing.

The `contributing file`_ contains style guides and other useful guidelines for contributing to the project.

.. _contributing file: https://github.com/cbarrick/toys/tree/master/CONTRIBUTING.rst


License
=======

MIT License

Copyright (c) 2017 Chris Barrick

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
