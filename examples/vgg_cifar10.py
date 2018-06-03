from tabulate import tabulate

import toys
from toys.datasets import CIFAR10
from toys.metrics import MultiMetric
from toys.model_selection import GridSearchCV
from toys.networks import VGG11
from toys.supervised import GradientDescent


train_data = CIFAR10(test=False)
test_data = CIFAR10(test=True)

grid_search = GridSearchCV(
    estimator = GradientDescent(
        ctor = VGG11,
        in_shape = (32, 32, 3),
        out_shape = 10,
        classifier = True,
        loss_fn = 'cross_entropy',
        optimizer = 'SGD:lr=1e-4',
        max_epochs = 200,
        batch_size = 32,
    ),
    param_grid = dict(
        activation = ['relu', 'elu'],
    ),
    metric = 'f_score',
    cv = 3,
)

estimator = grid_search(train_data)
print(tabulate(estimator.cv_results, headers='keys'))

model = estimator(train_data)
scorer = MultiMetric(('accuracy', 'f_score'))
accuracy, f_score = scorer(model, test_data)
print(f'Test accuracy: {accuracy:%}')
print(f'Test f-score: {f_score:%}')
