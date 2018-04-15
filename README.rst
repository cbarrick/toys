================================================================================
                    Toys: Friendly and Fast Machine Learning
================================================================================


Principals
==========
- **Familiarity**: Users of Scikit-learn should feel at home writing experiments
  with Toys. We aim to provide similar abstractions for quickly developing
  machine learning experiments, e.g. ``GridSearchCV``, with APIs which read
  almost the same as their Scikit counterparts. That being said, familiarity is
  not compatibility. The underlying design patterns employed by the Toys API is
  fundamentally different than that of Scikit-learn, and we are not afraid to
  break compatibility for the sake of a better developer experience.
- **GPU acceleration**: Graphics processing units have quickly become an
  important hardware accelerator in machine learning. Most Toys estimators
  support GPUs as first class hardware by leveraging PyTorch as the underlying
  computation library.
- **Deep Learning**: Deep learning techniques have lead to state of the art
  results in a variety of important machine learning domains, including
  computer vision and natural language processing. The Toys project provides
  training algorithms, like ``GradientDescent``, which wrap deep learning
  architectures implemented as PyTorch modules. Further, the ``toys.layers`` and
  ``toys.networks`` packages implement higher level layers (like ``Vgg`` blocks)
  and complete implementations for common architectures. The Toys API is
  designed to pick up where PyTorch leaves off.
- **Out of Core**: It is unreasonable to assume that our can fit into core
  memory. Rather than loading all of the data into numpy arrays, Toys borrows
  the notion of a ``Dataset`` from PyTorch. Your dataset need only have a
  length (``__len__``) and support access by index (``__getitem__``), which
  may dynamically load data from disk if desired. Existing small datasets need
  not change: if your data is already a list or array, it already implements
  the ``Dataset`` interface!


Toys vs Scikit-learn
====================
**TODO** Write this up:

- Both are very similar. Both solve the same problems with similar abstractions.
- Scikit-learn follows the prototype design pattern and conflates the estimator
  and model abstractions into the same classes. This makes estimator
  implementations highly prescriptive, introduces a non-trivial cloning
  semantics, and forces Scikit-learn specific naming conventions.
- Toys follows the factory design pattern (like Spark's MLLib); estimators are
  factories for models. Furthermore, every abstraction in Toys are implemented
  as callables rather than classes. This means that every role can be filled by
  simple functions or complex classes. See this talk `Stop Writing Classes`_ by
  Python core developer Jack Diederich.

.. _Stop Writing Classes: https://www.youtube.com/watch?v=o9pEzgHorH0


Development Roadmap
===================
While the above descriptions are written in the past tense, the actual
implementation of Toys is just getting started. The initial efforts are being
put towards implementing training algorithms which wrap PyTorch modules and
implementing model selection algorithms and infrastructure like ``GridSearchCV``
and scoring metrics. To quickly reach feature parity, we can wrap most
Scikit-learn estimators to the Toys API (they are quite similar), then replace
them one-by-one with PyTorch based implementations. In the long run, we'd like
to support Dask arrays for compatibility with that ecosystem of out-of-core
datasets and distributed computation, but currently there are upstream
compatibility issues between Dask and PyTorch.
