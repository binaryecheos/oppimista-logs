import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import numpy as np
from optim import GradientDescent
from loss import LossFunction


class LinkFunction:
    """Common link functions and their inverses."""

    # Identity link (Gaussian / linear regression)
    @staticmethod
    def identity_eta(mu):
        return mu

    @staticmethod
    def identity_mu(eta):
        return eta

    # Logit link (Bernoulli / logistic regression)
    @staticmethod
    def logit_eta(mu):
        mu = np.clip(mu, 1e-12, 1 - 1e-12)
        return np.log(mu / (1 - mu))

    @staticmethod
    def logit_mu(eta):
        return LossFunction.sigmoid(eta)

    # Log link (Poisson regression)
    @staticmethod
    def log_eta(mu):
        mu = np.clip(mu, 1e-12, None)
        return np.log(mu)

    @staticmethod
    def log_mu(eta):
        return np.exp(eta)


class GLM:
    """
    Generalized Linear Model (scalar responses).

    - Linear predictor: eta = X_b @ theta
    - Mean response:    mu = link_mu(eta)
    - Loss & gradient are provided from loss.py

    Handles:
        - Gaussian GLM (linear regression)
        - Bernoulli-logit GLM (binary logistic regression)
        - Poisson-log GLM (count regression)
    """

    def __init__(
        self,
        link_mu,          # eta -> mu
        loss_fn,          # (y_pred, y_true) -> scalar
        loss_grad_fn,     # (y_pred_logits, y_true, Xb) -> grad_theta
        lr=0.01,
        n_iters=1000,
        method="batch",
        batch_size=1,
        tol=1e-6,
    ):
        self.link_mu = link_mu
        self.loss_fn = loss_fn
        self.loss_grad_fn = loss_grad_fn
        self.lr = lr
        self.n_iters = n_iters
        self.method = method
        self.batch_size = batch_size
        self.tol = tol
        self.theta = None

    @staticmethod
    def _add_bias(X):
        n = X.shape[0]
        return np.c_[np.ones((n, 1)), X]

    def _eta(self, X):
        Xb = self._add_bias(X)
        return Xb @ self.theta

    def predict_mean(self, X):
        eta = self._eta(X)
        return self.link_mu(eta)

    def predict(self, X, threshold=0.5):
        mu = self.predict_mean(X)
        # If looks like probabilities, return hard labels
        if mu.ndim == 1 and (mu.min() >= 0) and (mu.max() <= 1):
            return (mu >= threshold).astype(int)
        return mu

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        if self.method == "batch":
            self.theta = GradientDescent.batch(
                X,
                y,
                loss_fn=self.loss_fn,
                loss_grad_fn=self.loss_grad_fn,
                lr=self.lr,
                n_iters=self.n_iters,
                tol=self.tol,
            )
        elif self.method == "sgd":
            self.theta = GradientDescent.sgd(
                X,
                y,
                loss_fn=self.loss_fn,
                loss_grad_fn=self.loss_grad_fn,
                lr=self.lr,
                n_iters=self.n_iters,
                batch_size=self.batch_size,
                tol=self.tol,
            )
        else:
            raise ValueError("method must be 'batch' or 'sgd'")
        return self


# ---------- scalar-response GLM factories ----------

def GaussianGLM(lr=0.01, n_iters=1000, method="batch", batch_size=1, tol=1e-6):
    """GLM for ordinary least squares (Gaussian, identity link)."""

    def loss_grad(y_pred_logits, y_true, Xb):
        return LossFunction.squared_error_gradient(y_pred_logits, y_true, Xb)

    return GLM(
        link_mu=LinkFunction.identity_mu,
        loss_fn=LossFunction.squared_error,
        loss_grad_fn=loss_grad,
        lr=lr,
        n_iters=n_iters,
        method=method,
        batch_size=batch_size,
        tol=tol,
    )


def BernoulliLogitGLM(lr=0.1, n_iters=1000, method="batch", batch_size=1, tol=1e-6):
    """GLM for binary logistic regression (Bernoulli, logit link)."""

    def loss_grad(y_pred_logits, y_true, Xb):
        p = LossFunction.sigmoid(y_pred_logits)
        return LossFunction.logistic_gradient(p, y_true, Xb)

    return GLM(
        link_mu=LinkFunction.logit_mu,
        loss_fn=LossFunction.logistic_loss,
        loss_grad_fn=loss_grad,
        lr=lr,
        n_iters=n_iters,
        method=method,
        batch_size=batch_size,
        tol=tol,
    )


def PoissonLogGLM(lr=0.01, n_iters=1000, method="batch", batch_size=1, tol=1e-6):
    """
    GLM for Poisson regression (counts).

    - Response: y in {0,1,2,...}
    - Mean:     mu = exp(eta)
    - Link:     log(mu) = eta
    - Loss:     negative Poisson log-likelihood (implemented in loss.py)
    """

    def loss_grad(y_pred_logits, y_true, Xb):
        # y_pred_logits = eta; mean mu = exp(eta)
        mu = LinkFunction.log_mu(y_pred_logits)
        # For Poisson, grad of negative log-likelihood:
        # grad = (1/n) Xᵀ (mu - y)
        n = Xb.shape[0]
        diff = mu - y_true
        return Xb.T @ diff / n

    return GLM(
        link_mu=LinkFunction.log_mu,
        loss_fn=LossFunction.poisson_loss,    # add in loss.py
        loss_grad_fn=loss_grad,
        lr=lr,
        n_iters=n_iters,
        method=method,
        batch_size=batch_size,
        tol=tol,
    )


# ---------- Multinomial / softmax regression (vector response) ----------

class SoftmaxGLM:
    """
    Multinomial logistic regression (a GLM for categorical y).

    - Parameters: Theta of shape (d+1, K)  (bias + d features, K classes)
    - Softmax:   P(y=k | x) = exp(eta_k) / sum_j exp(eta_j),
                 eta = X_b @ Theta
    - Loss:      multiclass cross-entropy
    """

    def __init__(self, lr=0.1, n_iters=1000, method="batch", batch_size=32, tol=1e-6):
        self.lr = lr
        self.n_iters = n_iters
        self.method = method
        self.batch_size = batch_size
        self.tol = tol
        self.Theta = None  # (d+1, K)

    @staticmethod
    def _add_bias(X):
        n = X.shape[0]
        return np.c_[np.ones((n, 1)), X]

    @staticmethod
    def _softmax(logits):
        # logits: (n, K)
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)

    def predict_proba(self, X):
        Xb = self._add_bias(X)
        logits = Xb @ self.Theta   # (n, K)
        return self._softmax(logits)

    def predict(self, X):
        probs = self.predict_proba(X)
        return probs.argmax(axis=1)

    def _loss_and_grad(self, Xb, y):
        """
        Cross-entropy loss and gradient for softmax regression.

        Xb: (n, d+1)
        y:  (n,) integer class labels in {0,...,K-1}
        """
        n = Xb.shape[0]
        logits = Xb @ self.Theta          # (n, K)
        probs = self._softmax(logits)     # (n, K)

        # one-hot
        K = self.Theta.shape[1]
        Y_onehot = np.zeros((n, K))
        Y_onehot[np.arange(n), y] = 1

        # cross-entropy
        eps = 1e-12
        loss = -np.sum(Y_onehot * np.log(probs + eps)) / n

        # gradient: (d+1, K)
        grad = Xb.T @ (probs - Y_onehot) / n
        return loss, grad

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        Xb = self._add_bias(X)

        n, d1 = Xb.shape
        num_classes = int(y.max()) + 1
        self.Theta = np.zeros((d1, num_classes))

        if self.method == "batch":
            for _ in range(self.n_iters):
                loss, grad = self._loss_and_grad(Xb, y)
                self.Theta -= self.lr * grad
        elif self.method == "sgd":
            for _ in range(self.n_iters):
                perm = np.random.permutation(n)
                for i in range(0, n, self.batch_size):
                    idx = perm[i:i + self.batch_size]
                    loss, grad = self._loss_and_grad(Xb[idx], y[idx])
                    self.Theta -= self.lr * grad
        else:
            raise ValueError("method must be 'batch' or 'sgd'")

        return self


# ---------- quick sanity checks ----------

if __name__ == "__main__":
    np.random.seed(42)

    # 1) Gaussian GLM ~ linear regression
    X_lin = np.random.randn(100, 1)
    theta_true_lin = np.array([1.0, 3.0])
    y_lin = (np.c_[np.ones((100, 1)), X_lin] @ theta_true_lin
             + 0.1 * np.random.randn(100))

    glm_gauss = GaussianGLM(lr=0.05, n_iters=500)
    glm_gauss.fit(X_lin, y_lin)
    print("Gaussian GLM theta:", glm_gauss.theta)

    # 2) Bernoulli-logit GLM ~ binary logistic regression
    X_log = np.random.randn(200, 2)
    theta_true_log = np.array([-0.5, 2.0, -1.0])
    Xb_log = np.c_[np.ones((200, 1)), X_log]
    p_true = LossFunction.sigmoid(Xb_log @ theta_true_log)
    y_log = (p_true > 0.5).astype(int)

    glm_bern = BernoulliLogitGLM(lr=0.1, n_iters=1000)
    glm_bern.fit(X_log, y_log)
    y_pred_log = glm_bern.predict(X_log)
    print("Bernoulli-logit GLM theta:", glm_bern.theta)
    print("Bernoulli-logit GLM acc:", (y_pred_log == y_log).mean())

    # 3) Poisson GLM ~ count regression
    X_pois = np.random.randn(200, 1)
    theta_true_pois = np.array([0.2, 0.5])
    lam = np.exp(np.c_[np.ones((200, 1)), X_pois] @ theta_true_pois)
    y_pois = np.random.poisson(lam)

    glm_pois = PoissonLogGLM(lr=0.01, n_iters=1000)
    glm_pois.fit(X_pois, y_pois)
    mu_pred = glm_pois.predict_mean(X_pois)
    print("Poisson GLM theta:", glm_pois.theta)
    print("Poisson GLM mean y (true, pred):", y_pois.mean(), mu_pred.mean())

    # 4) Softmax GLM ~ multiclass logistic regression
    n = 300
    K = 3
    X_soft = np.random.randn(n, 2)
    # simple synthetic classes
    y_soft = np.argmax(
        np.c_[X_soft[:, 0] + 0.5,
              -X_soft[:, 0] + X_soft[:, 1],
              -X_soft[:, 1] - 0.2],
        axis=1,
    )

    softmax_glm = SoftmaxGLM(lr=0.1, n_iters=500, method="batch")
    softmax_glm.fit(X_soft, y_soft)
    y_soft_pred = softmax_glm.predict(X_soft)
    print("Softmax GLM acc:", (y_soft_pred == y_soft).mean())