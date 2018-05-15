from tabulate import tabulate

import toys
from toys.datasets import CIFAR10
from toys.gradient_descent import GradientDescent
from toys.layers.networks import VGG11
from toys.metrics import MultiMetric
from toys.model_selection import GridSearchCV


train_data = CIFAR10(test=False)
test_data = CIFAR10(test=True)

metric = MultiMetric(
    metrics = ('accuracy', 'f_score'),
    batch_size = 256,
)

grid_search = GridSearchCV(
    estimator = GradientDescent(VGG11),
    param_grid = dict(
        activation = ['relu', 'elu'],
    ),
    in_shape = (32, 32, 3),
    out_shape = 10,
    classifier = True,
    loss_fn = 'cross_entropy',
    optimizer = 'SGD:lr=1e-4',
    max_epochs = 200,
    batch_size = 32,
    cv = 3,
    shuffle = True,
    metric = metric,
)

estimator = grid_search(train_data)
print(tabulate(estimator.cv_results, headers='keys'))

model = estimator(train_data)
accuracy, f_score = metric(model, test_data)
print(f'Test accuracy: {accuracy:%}')
print(f'Test f-score: {f_score:%}')
