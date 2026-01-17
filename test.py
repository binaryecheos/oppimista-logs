import numpy as np
from optim import GradientDescent
from loss import LossFunction

np.random.seed(42)

# toy binary data
n = 200
X = np.random.randn(n, 2)
true_theta = np.array([-0.5, 2.0, -1.0])

Xb = np.c_[np.ones((n, 1)), X]
logits = Xb @ true_theta
probs = LossFunction.sigmoid(logits)
y = (probs > 0.5).astype(int)

# Newton's method
theta_newton = GradientDescent.newton_logistic(X, y, n_iters=10)

# Compare with GD
theta_gd = GradientDescent.batch(
    X,
    y,
    loss_fn=LossFunction.logistic_loss,
    loss_grad_fn=lambda y_pred, y_true, Xb:
        LossFunction.logistic_gradient(
            LossFunction.sigmoid(y_pred), y_true, Xb
        ),
    lr=0.1,
    n_iters=1000,
)

print("True θ   :", true_theta)
print("Newton θ :", theta_newton)
print("GD θ     :", theta_gd)

def add_bias(X):
    return np.c_[np.ones((X.shape[0], 1)), X]

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def predict(theta, X, threshold=0.5):
    Xb = add_bias(X)
    p = sigmoid(Xb @ theta)
    return (p >= threshold).astype(int)

print("Acc true θ   :", (predict(true_theta, X) == y).mean())
print("Acc GD θ     :", (predict(theta_gd, X) == y).mean())
print("Acc Newton θ :", (predict(theta_newton, X) == y).mean())
