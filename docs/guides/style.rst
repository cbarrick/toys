Style Guide
==================================================

All Python code should follow the `Google Python Style Guide`_ with the following exceptions and additions.

.. _Google Python Style Guide: https://google.github.io/styleguide/pyguide.html


Doc strings
--------------------------------------------------

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
--------------------------------------------------

Type hints are useful for both documentation and static analysis tooling but can be very distracting syntactically. As a compromise, always include `PEP 484`_ compliant type hints in docstrings for arguments, return, and yield values. Don't include type annotations in code.

The following sugar is allowed, given in order of precedence:

- ``Union[A, B]`` may be written as ``A or B``.
- ``Callable[A, B]`` may be written as ``A -> B``.

Note that ``Optional[T]`` is equivalent to ``Union[T, None]``. The preferred notation for optional types is ``T or None``.

When types become complex, create an alias, e.g.::

    CrossValSplitter = Callable[[Dataset], Iterable[Tuple[Dataset, Dataset]]]

.. _Pep 484: https://www.python.org/dev/peps/pep-0484/


Imports
--------------------------------------------------

Use relative imports for anything under the current package, and use absolute imports for everything else. This allows package to be moved without modifying their contents, in the common case (other cases are a bad smell).

Import classes directly, using ``from pkg import MyClass``.

Group imports by dependency, and separate each group by a single blank line. The first import should be the top-level package of the dependency. Sort groups by dependency name except when conflicting with the following.

Reserve the first group for the Python standard library.

Reserve the second group for the SciPy stack, e.g. ``numpy``, ``scipy``, ``matplotlib``, and ``pandas``. Other general purpose data handling tools may be included in this section, like ``dask`` and ``xarray``. Use simple ``import`` statements in this group. If you find yourself writing many import from the same package, use a dedicated group instead.

Place relative imports last in their own group.

Within each group, sort bare ``import ...`` statements before ``from ... import ...`` statements. Otherwise sort imports lexicographically.

Always import the top-level package for each dependency. Import all objects used in docstrings, and use objects in docstrings as imported. Otherwise avoid dead imports.

E.g.::

    from typing import Any, Mapping, Sequence

    import numpy as np
    import scipy
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import pandas as pd

    import torch
    from torch.nn import DataParallel, Module
    from torch.optim import Optimizer

    import toys
    from toys import Dataset
    from toys.metrics import Mean

    from .cross_val import KFold


Code layout
--------------------------------------------------

Code is divided into packages (folders) and modules (\*.py files). By default, all code in modules is considered private. Public objects should be reexported by the package's ``__init__.py`` file. Other than comments and a package-level docstring, each ``__init__.py`` file should only contain relative import statements for the public objects in submodules of the package.

Do not use ``__all__``. The rules above serve the same purpose.
