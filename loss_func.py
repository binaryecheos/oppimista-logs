import numpy as np

class LossFunction:
    """General loss functions"""
    
    @staticmethod
    def squared_error(y_pred, y_true):
        """
        MSE: J(θ) = (1 / 2n) * Σ (y_pred - y_true)^2
        """
        y_pred = np.asarray(y_pred)
        y_true = np.asarray(y_true)
        diff = y_pred - y_true
        n = diff.size
        return 0.5 * np.sum(diff ** 2) / n
    
    @staticmethod
    def squared_error_gradient(y_pred, y_true, X):
        """
        ∇J(θ) = (1/n) * Xᵀ (Xθ - y)
        """
        n = X.shape[0]
        diff = y_pred - y_true
        return X.T @ diff / n