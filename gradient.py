from loss_func import LossFunction
import numpy as np

class GradientDescent:
    """General optimizer for any model"""
    
    @staticmethod
    def batch(X, y, loss_fn, loss_grad_fn, lr=0.01, n_iters=1000, tol=1e-6):
        """Batch GD"""
        X = GradientDescent._add_bias(X)
        theta = np.zeros(X.shape[1])
        
        for _ in range(n_iters):
            theta_old = theta.copy()
            y_pred = X @ theta
            grad = loss_grad_fn(y_pred, y, X)
            theta -= lr * grad
            
            if np.linalg.norm(theta - theta_old) < tol:
                break
        
        return theta
    
    @staticmethod
    def sgd(X, y, loss_fn, loss_grad_fn, lr=0.01, n_iters=1000, batch_size=1, tol=1e-6):
        """Stochastic/Mini-batch GD"""
        X = GradientDescent._add_bias(X)
        theta = np.zeros(X.shape[1])
        n_samples = X.shape[0]
        
        for iteration in range(n_iters):
            theta_old = theta.copy()
            
            # Shuffle
            indices = np.random.permutation(n_samples)
            
            # Mini-batches
            for i in range(0, n_samples, batch_size):
                batch_idx = indices[i:i+batch_size]
                X_b, y_b = X[batch_idx], y[batch_idx]
                
                y_pred = X_b @ theta
                grad = loss_grad_fn(y_pred, y_b, X_b)
                theta -= lr * grad
            
            if np.linalg.norm(theta - theta_old) < tol:
                break
        
        return theta
    
    @staticmethod
    def _add_bias(X):
        n = X.shape[0]
        return np.c_[np.ones((n, 1)), X]