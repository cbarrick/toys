================================================================================
                    Toys: Friendly and Fast Machine Learning
================================================================================

Toys is a framework for machine learning experiments inspired by Scikit-learn and built with PyTorch.


Principals
==========
- **Familiarity**: Users of Scikit-learn should feel at home writing experiments with Toys. We aim to provide similar abstractions for quickly developing machine learning experiments, e.g. ``GridSearchCV``, with APIs which read almost the same as their Scikit counterparts. That being said, familiarity is not compatibility. The underlying design patterns employed by the Toys API is fundamentally different than that of Scikit-learn, and we are not afraid to break compatibility for the sake of a better developer experience.
- **GPU acceleration**: Graphics processing units have quickly become an important hardware accelerator in machine learning. Most Toys estimators support GPUs as first class hardware by leveraging PyTorch as the underlying computation library.
- **Deep Learning**: Deep learning techniques have lead to state of the art results in a variety of important machine learning domains, including computer vision and natural language processing. The Toys project provides training algorithms, like ``GradientDescent``, which wrap deep learning architectures implemented as PyTorch modules. Further, the ``toys.layers`` and ``toys.networks`` packages implement higher level layers (like ``VGG`` blocks) and complete implementations for common architectures. The Toys API is designed to pick up where PyTorch leaves off.
- **Out of Core**: It is unreasonable to assume that our can fit into core memory. Rather than loading all of the data into numpy arrays, Toys borrows the notion of a ``Dataset`` from PyTorch. Your dataset need only have a length (``__len__``) and support access by index (``__getitem__``), which may dynamically load data from disk if desired. Existing small datasets need not change: if your data is already a list or array, it already implements the ``Dataset`` interface!


Toys vs Scikit-learn
====================
Scikit-learn is wonderful to use and provides a wealth of classic machine learning algorithms. Other frameworks like Weka and Spark pale in comparison in terms of breadth and flexibility. Unfortunately, Scikit-learn committed to architectural designs early on that limit it today. Specifically, Scikit-learn cannot support out-of-core learning (datasets are assumed to be numpy arrays), deep learning and probabilistic graphical models (both require large frameworks in and of themselves), reinforcement learning, or GPU acceleration. Additionally, Scikit-learn has a few historical warts, like the conflation of estimators and models, that makes writing new compatible algorithm more painful that it should be.

Toys is an effort to rectify this situation by providing a similarly useful API built on a more modern and flexible software stack. Specifically, Toys uses `PyTorch`_, rather than Scipy, as the underlying computation layer. This gives us GPU acceleration for free and accommodates out of core datasets for iterative, batch oriented algorithms. Building on PyTorch gives us access to the requisite vocabulary for deep learning (PyTorch includes a neural net library and automatic differentiator), and the `Pyro`_ library provides a compatible framework for probabilistic models. To design a machine learning framework from scratch gives us the opportunity to accommodate a wider range of tasks, including reinforcement learning through the now de facto standard API of OpenAI's `Gym`_. In other words, the state of machine learning in Python is much broader, more developed, and more computationally demanding than it was in 2007 when Scikit-learn was started. Toys embraces these changes and strives to be fast, friendly, and universal.

Toys aims to provide an API that is remarkably similar to Scikit-learn, but the underlying architecture is fundamentally different. In Scikit-learn, estimators, predictors, transformers, and models are `conceptually different things <http://scikit-learn.org/stable/developers/contributing.html#apis-of-scikit-learn-objects>`_, but in practice all concepts are implemented by a single class; each concept corresponds to a method of that class. This leads to a prototype pattern needed to support model selection and validation, and along with this pattern come strict rules about when the class is allowed to do useful work, a required naming convention for learned parameters, and a non-trivial cloning semantics (and the ``set_params`` method required to support it). On the other hand, Toys simplifies this situation into just two concepts, models and estimators. A model is simply a function which takes data and transforms it, typically into a prediction or a reduced feature space. Models are usually learned functions, and thus created dynamically. The factories for these models are called estimators. In other words, estimators are learning algorithms, and models are the functions they produce. Further, since models and estimators are just functions, they can be implemented as such. Of course, if you need the power of full blown classes, you are free to do so. Toys takes this functional design seriously; whenever possible, auxiliary components, like cross validation splitters, can be (and are) implemented as functions. For more motivation on this design, see the talk `Stop Writing Classes`_ by Python core developer Jack Diederich.

.. _Gym: https://gym.openai.com/
.. _Pyro: http://pyro.ai/
.. _PyTorch: http://pytorch.org/
.. _Stop Writing Classes: https://www.youtube.com/watch?v=o9pEzgHorH0


Development Roadmap
===================
While the above descriptions are very forward thinking, the actual implementation of Toys is just getting started. My initial effort is being put towards implementing training algorithms which wrap PyTorch modules, implementing model selection algorithms like ``GridSearchCV``, and implementing the most important scoring metrics. In the long run, I'd like to support Dask arrays for compatibility with that ecosystem of out-of-core datasets and distributed computation, but currently there are upstream compatibility issues between Dask and PyTorch. The hope is to one day reach feature parity with Scikit-learn, but this will certainly require a healthy community of contributors. To reach that point, I first need to develop the infrastructure and design guidelines to foster distributed development.

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
- A tool to extract MyPy type annotation stubs from doc strings
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
