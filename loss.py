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
    
    @staticmethod
    def sigmoid(z):
        z = np.asarray(z)
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def logistic_loss(y_pred_proba, y_true):
        """
        Binary cross-entropy loss:

        J(θ) = -(1/n) Σ [ y log(p) + (1-y) log(1-p) ],
        where p = σ(θᵀx).

        y_pred_proba should already be probabilities in (0,1).
        """
        y_true = np.asarray(y_true)
        y_pred_proba = np.clip(y_pred_proba, 1e-12, 1 - 1e-12)
        n = y_true.size
        return -np.sum(
            y_true * np.log(y_pred_proba) +
            (1 - y_true) * np.log(1 - y_pred_proba)
        ) / n
    
    @staticmethod
    def logistic_gradient(y_pred_proba, y_true, X):
        """
        Gradient of binary cross-entropy loss w.r.t θ

        ∇J(θ) = (1/n) * Xᵀ (p - y)
        """
        y_true = np.asarray(y_true)
        y_pred_proba = np.asarray(y_pred_proba)
        n = X.shape[0]
        return X.T @ (y_pred_proba - y_true) / n

