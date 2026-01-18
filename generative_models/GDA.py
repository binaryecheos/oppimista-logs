import numpy as np


class GDA:
    """
    Gaussian Discriminant Analysis (binary).

    Assumes:
        y ∈ {0,1}
        x | y = k ~ N(μ_k, Σ)   with shared covariance Σ

    Learns:
        φ      = P(y=1)
        μ0, μ1 = class means
        Σ      = shared covariance

    Predicts using the equivalent logistic form:
        p(y=1 | x) = sigmoid(thetaᵀ x_tilde)
    where x_tilde = [1, x] and theta is derived from generative params.
    """

    def __init__(self):
        self.phi = None       # P(y=1)
        self.mu0 = None
        self.mu1 = None
        self.Sigma = None
        self.theta = None     # [theta0, theta_vec...] for logistic view

    def fit(self, X, y):
        """
        X: (n, d)
        y: (n,) in {0,1}
        """
        X = np.asarray(X)
        y = np.asarray(y)
        n, d = X.shape

        # class priors
        self.phi = np.mean(y)         # P(y=1)
        n1 = y.sum()
        n0 = n - n1

        # class means
        X0 = X[y == 0]
        X1 = X[y == 1]
        self.mu0 = X0.mean(axis=0)
        self.mu1 = X1.mean(axis=0)

        # shared covariance
        Sigma = np.zeros((d, d))
        for i in range(n0):
            diff = (X0[i] - self.mu0).reshape(-1, 1)
            Sigma += diff @ diff.T
        for i in range(n1):
            diff = (X1[i] - self.mu1).reshape(-1, 1)
            Sigma += diff @ diff.T
        Sigma /= n
        self.Sigma = Sigma

        # precompute logistic-equivalent theta for faster prediction
        Sigma_inv = np.linalg.inv(Sigma)
        theta_vec = Sigma_inv @ (self.mu1 - self.mu0)

        # bias term (theta0)
        mu0_term = self.mu0.T @ Sigma_inv @ self.mu0
        mu1_term = self.mu1.T @ Sigma_inv @ self.mu1
        theta0 = -0.5 * (mu1_term - mu0_term) + np.log(self.phi / (1 - self.phi))

        self.theta = np.concatenate([[theta0], theta_vec])
        return self

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    def predict_proba(self, X):
        """
        Returns P(y=1 | x) using the logistic form derived from GDA.
        """
        X = np.asarray(X)
        n = X.shape[0]
        Xb = np.c_[np.ones((n, 1)), X]
        logits = Xb @ self.theta
        return self._sigmoid(logits)

    def predict(self, X, threshold=0.5):
        p = self.predict_proba(X)
        return (p >= threshold).astype(int)


# ---------- Multiclass GDA (one Gaussian per class, shared Σ) ----------

class GDA_Multiclass:
    """
    Multiclass GDA with shared covariance matrix.

    Assumes:
        y ∈ {0,1,...,K-1}
        x | y = k ~ N(μ_k, Σ) with shared Σ.

    Uses Bayes rule:
        p(y=k | x) ∝ p(y=k) * N(x; μ_k, Σ)
    """

    def __init__(self):
        self.phi = None      # (K,) class priors
        self.mu = None       # (K, d) class means
        self.Sigma = None    # (d, d) shared covariance

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        n, d = X.shape
        classes = np.unique(y)
        K = len(classes)

        self.mu = np.zeros((K, d))
        self.phi = np.zeros(K)
        Sigma = np.zeros((d, d))

        for k_idx, k in enumerate(classes):
            Xk = X[y == k]
            nk = Xk.shape[0]
            self.phi[k_idx] = nk / n
            self.mu[k_idx] = Xk.mean(axis=0)

        # shared covariance
        for k_idx, k in enumerate(classes):
            Xk = X[y == k]
            for i in range(Xk.shape[0]):
                diff = (Xk[i] - self.mu[k_idx]).reshape(-1, 1)
                Sigma += diff @ diff.T
        Sigma /= n
        self.Sigma = Sigma
        return self

    def _log_gaussian(self, X):
        """
        Compute log N(x; μ_k, Σ) for all k and all x.

        Returns:
            log_prob: shape (n, K)
        """
        X = np.asarray(X)
        n, d = X.shape
        K = self.mu.shape[0]

        Sigma_inv = np.linalg.inv(self.Sigma)
        det_Sigma = np.linalg.det(self.Sigma)
        const = -0.5 * (d * np.log(2 * np.pi) + np.log(det_Sigma))

        log_prob = np.zeros((n, K))
        for k in range(K):
            diff = X - self.mu[k]
            # each row: (x - μ_k)^T Σ^{-1} (x - μ_k)
            quad = np.sum(diff @ Sigma_inv * diff, axis=1)
            log_prob[:, k] = const - 0.5 * quad
        return log_prob

    def predict_proba(self, X):
        """
        Returns class probabilities p(y=k | x).
        """
        X = np.asarray(X)
        log_px_y = self._log_gaussian(X)            # (n, K)
        log_prior = np.log(self.phi + 1e-12)        # (K,)
        log_joint = log_px_y + log_prior           # (n, K)

        # softmax in log-space
        log_joint -= log_joint.max(axis=1, keepdims=True)
        exp_joint = np.exp(log_joint)
        probs = exp_joint / exp_joint.sum(axis=1, keepdims=True)
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return probs.argmax(axis=1)


# ---------- quick sanity check ----------

if __name__ == "__main__":
    np.random.seed(42)

    # Binary GDA vs logistic-style data
    n = 400
    d = 2
    X = np.random.randn(n, d)
    theta_true = np.array([-0.3, 1.0, -0.8])
    Xb = np.c_[np.ones((n, 1)), X]
    logits = Xb @ theta_true
    p = 1 / (1 + np.exp(-logits))
    y = (p > 0.5).astype(int)

    gda = GDA()
    gda.fit(X, y)
    y_pred = gda.predict(X)
    acc = (y_pred == y).mean()
    print("Binary GDA acc:", acc)

    # Multiclass GDA sanity check
    n = 600
    K = 3
    d = 2
    X_mc = np.zeros((n, d))
    y_mc = np.zeros(n, dtype=int)
    points_per_class = n // K

    for k in range(K):
        mean = np.array([2 * np.cos(2 * np.pi * k / K),
                         2 * np.sin(2 * np.pi * k / K)])
        cov = np.array([[1.0, 0.3],
                        [0.3, 1.0]])
        X_mc[k * points_per_class:(k + 1) * points_per_class] = \
            np.random.multivariate_normal(mean, cov, size=points_per_class)
        y_mc[k * points_per_class:(k + 1) * points_per_class] = k

    gda_mc = GDA_Multiclass()
    gda_mc.fit(X_mc, y_mc)
    y_mc_pred = gda_mc.predict(X_mc)
    acc_mc = (y_mc_pred == y_mc).mean()
    print("Multiclass GDA acc:", acc_mc)


