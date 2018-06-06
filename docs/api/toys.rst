toys
==================================================
.. automodule:: toys


Core protocols
--------------------------------------------------

The core protocols are pure abstract classes. They provide no functionality and are for documentation purpose only. There is no requirement to subclass them; however doing so provides certain runtime protections through Python's abstract base class (:mod:`abc`) functionality.

.. autosummary::
	:toctree: stubs

	Dataset
	Estimator
	Model
	Metric


Common classes
--------------------------------------------------

.. autosummary::
	:toctree: stubs

	BaseEstimator
	TunedEstimator
	TorchModel


Dataset utilities
--------------------------------------------------

.. autosummary::
	:toctree: stubs

	toys.batches
	toys.zip
	toys.concat
	toys.flatten
	toys.subset
	toys.shape


Type aliases
--------------------------------------------------

These type aliases exist to aid in documentation and static analysis. They are irrelevant at runtime.


Shapes
^^^^^^^^^^^^^^^^^^^^^^^^^

.. class:: toys.ColumnShape
 	:annotation: = Optional[Tuple[Optional[int], ...]]

	The shape of a single datum in a column. :obj:`None` is used for dimensions of variable length, and when the total number of dimensions is variable.

	Note that the shape of a column *does not include the index dimension*.


.. class:: toys.RowShape
 	:annotation: = Optional[Tuple[ColumnShape, ...]]

	The shape of a row is the sequence of (possibly variable) shapes of its columns. The dataset shape may be :obj:`None` to indicate that the number of columns is variable.


For example, the :class:`~toys.datasets.CIFAR10` dataset has two columns. The first contains 32x32 RGB images; it's shape is ``(32, 32, 3)``. The second contains scalar class labels; it's shape is ``()``. The shape of the whole row is thus ``((32, 32, 3), ())``.

>>> from toys.datasets import CIFAR10
>>> cifar = CIFAR10()
>>> toys.shape(cifar)
((32, 32, 3), ())


Model selection
^^^^^^^^^^^^^^^^^^^^^^^^^
.. class:: CrossValSplitter
 	:annotation: = Callable[[Dataset], Iterable[Fold]]

	A function that takes a dataset and returns an iterable over some :class:`Fold` s of the dataset. These can be used by meta-estimators, like :class:`~toys.model_selection.GridSearchCV`, to test how estimators generalize to unseen data.


.. class:: Fold
 	:annotation: = Tuple[Dataset, Dataset]

	A fold is the partitioning of a dataset into two disjoint subsets, ``(train, test)``.


.. class:: ParamGrid
 	:annotation: = Mapping[str, Sequence]
