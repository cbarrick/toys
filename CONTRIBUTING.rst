================================================================================
                      Guidelines for contributing to Toys
================================================================================


Python Style
============
All Python code should follow the `Google Python Style Guide`_ with the following exceptions and additions.

.. _Google Python Style Guide: https://google.github.io/styleguide/pyguide.html

Doc strings
-----------
Doc strings should use triple single-quotes (``'''``).

All values in return, yield, attributes, arguments, and keyword arguments sections must include both names and type annotations. (See the following section on type annotations).

The description for arguments and return values start on the line following their name and type annotation. This is more visually appealing when long type annotations are used, and so we require it globally for consistency.

E.g.::

    def torch_dtype(dtype):
        '''Casts dtype to a PyTorch tensor class.

        The input may be a conventional name, like 'float' and 'double', or an
        explicit name like 'float32' or 'float64'. If the input is a known
        tensor class, it is returned as-is.

        Args:
            dtype (str or TorchDtype):
                A conventional name, explicit name, or known tensor class.

        Returns:
            cls (TorchDtype):
                The tensor class corresponding to `dtype`.
        '''
        ...

Type Annotations
----------------
Type hints are useful for both documentation and static analysis tooling but can be very distracting syntactically. As a compromise, always include `PEP 484`_ compliant type hints in doc strings for arguments, return, and yield values. Don't include type annotations in code.

The following sugar is allowed, given in order of precedence:

- ``Union[A, B]`` may be written as ``A or B``.
- ``Callable[A, B]`` may be written as ``A -> B``.

Note that ``Optional[T]`` is equivalent to ``Union[T, None]``. The preferred notation for optional types is ``T or None``.

When types become complex, create an alias, e.g.::

    CrossValSplitter = Callable[[Dataset], Iterable[Tuple[Dataset, Dataset]]]

.. _Pep 484: https://www.python.org/dev/peps/pep-0484/

Imports
-------
Imports within the same immediate package are relative. All other imports are absolute. This allows package names to be changed without modifying code within the package, in the common case.

Imports are grouped by top-level package, and each group is separated by a single blank line. Groups are sorted by top-level package name except when conflicting with the rules below.

The the first import group is reserved for the Python standard library.

The second import group is reserved for the "scientific Python standard library". That is ``numpy``, ``scipy``, ``matplotlib``, and ``pandas``. Other packages that may be included in this section include ``dask`` and ``xarray`` or any other general purpose data handling tool. All imports in this section must be simple. If you find yourself importing many things from any of these packages, use a dedicated import group instead.

Relative imports and imports from the ``toys`` package are the last and second to last groups respectively.

Groups for specific packages always import the top-level package directly as the first line of the group. All other lines are sorted.

All classes used in type signatures in the doc strings should be imported and in scope as used. This makes type signatures unambiguous. Otherwise dead imports are not allowed.

E.g.::

    from typing import Any, Mapping, Sequence
    import logging

    import numpy as np
    import scipy
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import pandas as pd

    import torch
    from torch.autograd import Variable
    from torch.nn import DataParallel, Module
    from torch.optim import Optimizer
    from torch.utils.data import DataLoader

    import toys
    from toys.accumulators import Mean
    from toys.datasets.utils import Dataset

    from .estimator import Estimator, Model
    from .torch import TorchModel, TorchDtype

Code layout
-----------

Code is divided into packages (folders) and modules (\*.py file). By default, all code in modules is considered private. Public objects should be reexported by the package's ``__init__.py`` file. Other than comments and a package-level doc string, each ``__init__.py`` file should only contain relative import statements, importing the public objects from other modules in the same package.
