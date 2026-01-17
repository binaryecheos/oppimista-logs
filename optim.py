from loss import LossFunction
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
    
    @staticmethod
    def newton_logistic(X, y, n_iters=20, tol=1e-6):
        """
        Newton's method for logistic regression.

        θ^(t+1) = θ^(t) - H^{-1} ∇J(θ^(t))
        where:
            ∇J(θ) = (1/n) Xᵀ (hθ(x) - y)
            H     = (1/n) Xᵀ R X
            R     = diag(p_i (1 - p_i)), p_i = σ(θᵀ x_i)

        X is (n, d) without bias; bias term is added inside.
        """
        Xb = GradientDescent._add_bias(X)     # (n, d+1)
        n, d1 = Xb.shape
        theta = np.zeros(d1)

        for _ in range(n_iters):
            theta_old = theta.copy()

            z = Xb @ theta
            p = LossFunction.sigmoid(z)

            # gradient: (d+1,)
            grad = Xb.T @ (p - y) / n

            # Hessian: (d+1, d+1)
            r = p * (1 - p)                   # shape (n,)
            R = np.diag(r)
            H = Xb.T @ R @ Xb / n

            # Newton step: solve H Δθ = grad
            step = np.linalg.solve(H, grad)
            theta -= step

            if np.linalg.norm(step) < tol:
                break

        return theta
    
    