import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


import numpy as np
from optim import GradientDescent
from loss import LossFunction

# 1. create sample data (binary classification)
np.random.seed(42)
n = 200
X = np.random.randn(n, 2)                 # features
true_theta = np.array([-0.5, 2.0, -1.0])  # bias, w1, w2

Xb = np.c_[np.ones((n, 1)), X]
logits = Xb @ true_theta
probs = 1 / (1 + np.exp(-logits))
y = (probs > 0.5).astype(int)             # labels in {0,1}

# 2. train logistic regression using shared GD + logistic loss
theta = GradientDescent.batch(
    X,
    y,
    loss_fn=LossFunction.logistic_loss,
    loss_grad_fn=lambda logits, y_true, Xb:
        LossFunction.logistic_gradient(
            LossFunction.sigmoid(logits),
            y_true,
            Xb
        ),
    lr=0.1,
    n_iters=1000,
)


print("Learned theta:", theta)

# 3. prediction helper
def predict(X, theta, threshold=0.5):
    Xb = np.c_[np.ones((X.shape[0], 1)), X]
    probs = 1 / (1 + np.exp(-(Xb @ theta)))
    return (probs >= threshold).astype(int)

y_pred = predict(X, theta)
print("Accuracy:", (y_pred == y).mean())
