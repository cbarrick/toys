from tabulate import tabulate

import toys
from toys.datasets import SimulatedLinear
from toys.metrics import MeanSquaredError
from toys.model_selection import GridSearchCV
from toys.supervised import LeastSquares


train_data = SimulatedLinear(length=100, seed='train')
test_data = SimulatedLinear(length=100, seed='test')

metric = MeanSquaredError()

cross_val = GridSearchCV(
    estimator = LeastSquares(),
    cv = 3,
    metric = metric,
    minimize = True,
)

estimator = cross_val(train_data)
print(tabulate(estimator.cv_results, headers='keys'))

model = estimator(train_data)
mse = metric(model, test_data)
print(f'Test set mean squared error: {mse:.2e}')
