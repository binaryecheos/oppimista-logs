import numpy as np
from gradient import GradientDescent
from loss_func import LossFunction

# Linear Regression
X = np.random.randn(100, 1)
y = 3 * X.flatten() + 1 + 0.1 * np.random.randn(100)

# Batch GD
theta_batch = GradientDescent.batch(
    X, y, 
    LossFunction.squared_error,
    LossFunction.squared_error_gradient,
    lr=0.01
)

# SGD
theta_sgd = GradientDescent.sgd(
    X, y,
    LossFunction.squared_error,
    LossFunction.squared_error_gradient,
    lr=0.01, batch_size=1
)

print("Batch GD theta:", theta_batch)
print("SGD theta:", theta_sgd)