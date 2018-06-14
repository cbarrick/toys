toys.model_selection
==================================================
.. automodule:: toys.model_selection


Functions
--------------------------------------------------

.. autosummary::
	:toctree: stubs

	combinations


Hyperparameter search
--------------------------------------------------

.. autosummary::
	:toctree: stubs

	GridSearchCV


Cross validation splitting
--------------------------------------------------

.. autosummary::
	:toctree: stubs

	KFold


Type aliases
--------------------------------------------------

These type aliases exist to aid in documentation and static analysis. They are irrelevant at runtime.

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
