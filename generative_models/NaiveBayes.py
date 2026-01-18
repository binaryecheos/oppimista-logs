import numpy as np


class GaussianNaiveBayes:
    """
    Gaussian Naive Bayes classifier.

    Assumes:
        - Features are continuous.
        - Conditional independence:
            x_j | y=k ~ N(μ_{k,j}, σ_{k,j}^2) independently across j.
    """

    def __init__(self):
        self.classes_ = None     # (K,)
        self.class_prior_ = None # (K,)
        self.mean_ = None        # (K, d)
        self.var_ = None         # (K, d)

    def fit(self, X, y):
        """
        X: (n, d)
        y: (n,)
        """
        X = np.asarray(X)
        y = np.asarray(y)
        n, d = X.shape

        classes = np.unique(y)
        K = len(classes)

        self.classes_ = classes
        self.class_prior_ = np.zeros(K)
        self.mean_ = np.zeros((K, d))
        self.var_ = np.zeros((K, d))

        for idx, c in enumerate(classes):
            Xc = X[y == c]
            self.class_prior_[idx] = Xc.shape[0] / n
            self.mean_[idx] = Xc.mean(axis=0)
            # add small epsilon for numerical stability
            self.var_[idx] = Xc.var(axis=0) + 1e-9

        return self

    def _log_gaussian(self, X):
        """
        Compute log p(x_j | y=k) under diagonal Gaussian for all k.

        Returns:
            log_prob: shape (n, K)
        """
        X = np.asarray(X)
        n, d = X.shape
        K = self.mean_.shape[0]

        log_prob = np.zeros((n, K))
        for idx in range(K):
            mean = self.mean_[idx]
            var = self.var_[idx]

            # per-feature Gaussian log-density, sum over j due to independence
            log_det = -0.5 * np.sum(np.log(2 * np.pi * var))
            quad = -0.5 * np.sum(((X - mean) ** 2) / var, axis=1)
            log_prob[:, idx] = log_det + quad
        return log_prob

    def predict_proba(self, X):
        X = np.asarray(X)
        log_px_y = self._log_gaussian(X)          # (n, K)
        log_prior = np.log(self.class_prior_ + 1e-12)  # (K,)
        log_joint = log_px_y + log_prior          # (n, K)

        # log-softmax for numerical stability
        log_joint -= log_joint.max(axis=1, keepdims=True)
        exp_joint = np.exp(log_joint)
        probs = exp_joint / exp_joint.sum(axis=1, keepdims=True)
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[probs.argmax(axis=1)]


class MultinomialNaiveBayes:
    """
    Multinomial Naive Bayes classifier.

    Typical for bag-of-words counts:
        x_j | y=k ~ Multinomial with parameters φ_{k,j}
    with independence across features given class.

    Works on non-negative integer feature vectors (counts).
    """

    def __init__(self, alpha=1.0):
        """
        alpha: Laplace smoothing parameter (α > 0).
        """
        self.alpha = alpha
        self.classes_ = None        # (K,)
        self.class_prior_ = None    # (K,)
        self.feature_log_prob_ = None  # (K, d)

    def fit(self, X, y):
        """
        X: (n, d) non-negative counts
        y: (n,)
        """
        X = np.asarray(X)
        y = np.asarray(y)
        n, d = X.shape

        classes = np.unique(y)
        K = len(classes)

        self.classes_ = classes
        self.class_prior_ = np.zeros(K)
        self.feature_log_prob_ = np.zeros((K, d))

        for idx, c in enumerate(classes):
            Xc = X[y == c]
            self.class_prior_[idx] = Xc.shape[0] / n

            # total counts per feature for this class
            class_count = Xc.sum(axis=0)  # (d,)

            # Laplace smoothing:
            # φ_{k,j} = (count_{k,j} + α) / (sum_j count_{k,j} + α * d)
            smoothed = class_count + self.alpha
            smoothed /= smoothed.sum()
            self.feature_log_prob_[idx] = np.log(smoothed + 1e-12)

        return self

    def predict_proba(self, X):
        """
        X: (n, d) non-negative counts
        Returns class probabilities using log-space computations.
        """
        X = np.asarray(X)
        n, d = X.shape
        K = self.feature_log_prob_.shape[0]

        log_prior = np.log(self.class_prior_ + 1e-12)  # (K,)
        # log-likelihood: sum_j x_j log φ_{k,j}
        log_likelihood = X @ self.feature_log_prob_.T  # (n, K)
        log_joint = log_likelihood + log_prior         # (n, K)

        log_joint -= log_joint.max(axis=1, keepdims=True)
        exp_joint = np.exp(log_joint)
        probs = exp_joint / exp_joint.sum(axis=1, keepdims=True)
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[probs.argmax(axis=1)]


# ---------- quick sanity checks ----------

if __name__ == "__main__":
    np.random.seed(42)

    # 1) Gaussian NB on synthetic continuous data
    n = 400
    d = 2
    X0 = np.random.multivariate_normal(mean=[-1, 0], cov=[[1.0, 0.2], [0.2, 1.0]], size=n // 2)
    X1 = np.random.multivariate_normal(mean=[1, 0.5], cov=[[1.2, -0.3], [-0.3, 1.0]], size=n // 2)
    X = np.vstack([X0, X1])
    y = np.array([0] * (n // 2) + [1] * (n // 2))

    gnb = GaussianNaiveBayes()
    gnb.fit(X, y)
    y_pred = gnb.predict(X)
    print("Gaussian NB acc:", (y_pred == y).mean())

    # 2) Multinomial NB on simple bag-of-words-like counts
    n = 300
    d = 5
    K = 3

    # class-specific word distributions
    phi_true = np.array([
        [0.6, 0.1, 0.1, 0.1, 0.1],   # class 0
        [0.1, 0.6, 0.1, 0.1, 0.1],   # class 1
        [0.1, 0.1, 0.1, 0.6, 0.1],   # class 2
    ])
    doc_length = 20

    X_counts = np.zeros((n, d), dtype=int)
    y_counts = np.zeros(n, dtype=int)

    for i in range(n):
        c = np.random.randint(0, K)
        y_counts[i] = c
        X_counts[i] = np.random.multinomial(doc_length, phi_true[c])

    mnb = MultinomialNaiveBayes(alpha=1.0)
    mnb.fit(X_counts, y_counts)
    y_counts_pred = mnb.predict(X_counts)
    print("Multinomial NB acc:", (y_counts_pred == y_counts).mean())
