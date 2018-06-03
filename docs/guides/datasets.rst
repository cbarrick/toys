Datasets
==================================================

The :class:`~toys.Dataset` protocol is borrowed from PyTorch and is the boundary between the preprocess and the model. The protocol is quite easy to implement. A dataset need only have methods :meth:`~object.__len__` and :meth:`~object.__getitem__` with integer indexing. Most simple collections can be used as datasets, including :class:`list` and :class:`~numpy.ndarray`.

We use the following vocabulary to describe datasets:

:Row: The value at ``dataset[i]`` is called the |ith| row of the dataset. Each row must be a sequence of arrays and/or scalars, and each array may be of different shape.

:Column: The positions in a row are called the columns. Columns are either **feature columns** or **target columns**.

:Supervised: A supervised dataset has at least two columns where the last column is a **target column** rather than a feature column.

:Feature: The data in any one feature column of a row is called a feature of that row.

:Target: Likewise, the data in the target column of a row is called the target of that row.

:Instance: The features of a row are collectively called an instance.

:Shape: The shape of a row or instance is the sequence of shapes of its columns. The shape of a dataset is the shape of its rows. Additionally, the shape of a dataset *does not include its length*.

For example, the :class:`~toys.datasets.CIFAR10` dataset is a supervised dataset with two columns. The feature column contains 32x32 pixel RGB images, and the target column contains integer class labels. The shape of the feature is ``(32, 32, 3)``, and the shape if the target is ``()`` (i.e. the target is a scalar). The shape of the CIFAR10 dataset is thus ``((32,32,3), ())``.

.. note::
    Unlike arrays, columns need not have the same shape across all rows. In fact, the same column may have a different number of dimensions in different rows, and the rows may even have different number of columns all together. While most estimators expect some consistency, this freedom allows us to efficiently represent, e.g., variable sequence lengths. A dataset shape (as opposed to a row or instance shape) may use :obj:`None` to represent a variable aspect of its shape.

.. |ith| replace:: i\ :sup:`th`


Creating and combining datasets
--------------------------------------------------

The primary functions for combining datasets are :func:`toys.concat` and :func:`toys.zip` which concatenate datasets by rows and columns respectively.

Of these, :func:`toys.zip` is the more commonly used. It allows us to, e.g., combine the features and target from separate datasets:

    >>> features = np.random.random(size=(100, 1, 5))  # 100 rows, 1 column of shape (5,)
    >>> target = np.prod(features, axis=-1)            # 100 rows, 1 scalar column
    >>> dataset = toys.zip(features, target)           # 100 rows, 2 columns
    >>> toys.shape(features)
    ((5,),)
    >>> toys.shape(target)
    ((),)
    >>> toys.shape(dataset)
    ((5,), ())

Most estimators will automatically zip datasets if you pass more than one:

    >>> from toys.supervised import LeastSquares
    >>> estimator = LeastSquares()
    >>> model = estimator(dataset)           # Each of these calls
    >>> model = estimator(features, target)  # is equivalent to the other


Batching and iteration
--------------------------------------------------

The function :func:`toys.batches` iterates over mini-batches of a dataset by delegating to PyTorch's :class:`~torch.utils.data.DataLoader` class. The :func:`~toys.batches` function forwards all of its arguments to the :class:`~torch.utils.data.DataLoader` constructor, but it allows the dataset to recommend default values through the :attr:`Dataset.hints` attribute. This allows the dataset to, e.g. specify an appropriate collate function or sampling strategy.

The most common arguments are:

:batch_size: The maximum number of rows per batch.

:shuffle: A boolean set to true to sample batches at random without replacement.

:collate_fn: A function to merge a list of samples into a mini-batch. This is required if the shape of the dataset is variable, e.g. to pad or pack a sequence length.

:pin_memory: If true, batches are loaded into CUDA pinned memory. Unlike vanilla PyTorch, this defaults to true whenever CUDA is available.

.. note::
	Most estimators will require an explicit ``batch_size`` argument when it can effect model performance. Thus the ``batch_size`` hint provided by the dataset is more influential to scoring functions than to estimators. Therefore the hinted value should be for scoring purposes and can be quite large.

.. seealso::
	See :class:`torch.utils.data.DataLoader` for a full description of all possible arguments.

.. todo::
	Add examples
