import numpy as np
from torch import erf

class Activations:
    """
    Collection of common neural network activation functions.
    All methods are static and work with scalars or NumPy arrays.
    """

    # ----- Basic activations -----

    @staticmethod
    def linear(x):
        return x

    @staticmethod
    def sigmoid(x):
        # f(x) = 1 / (1 + e^(-x)) 
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def tanh(x):
        # f(x) = (e^x - e^(-x)) / (e^x + e^(-x)) 
        return np.tanh(x)

    @staticmethod
    def relu(x):
        # f(x) = max(0, x) 
        return np.maximum(0, x)

    @staticmethod
    def leaky_relu(x, alpha=0.01):
        # f(x) = x if x>0 else alpha*x 
        return np.where(x > 0, x, alpha * x)

    @staticmethod
    def elu(x, alpha=1.0):
        # f(x) = x if x>0 else alpha*(e^x - 1) 
        return np.where(x > 0, x, alpha * (np.exp(x) - 1.0))

    @staticmethod
    def softplus(x):
        # f(x) = log(1 + e^x) [web:48]
        return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

    # ----- Probabilistic / output activations -----

    @staticmethod
    def softmax(x, axis=-1):
        """
        Numerically stable softmax along given axis. [web:45]
        """
        x = np.asarray(x)
        x_shifted = x - np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    # ----- Modern activations -----

    @staticmethod
    def swish(x, beta=1.0):
        # f(x) = x * sigmoid(beta * x) 
        return x * Activations.sigmoid(beta * x)

    @staticmethod
    def gelu(x, approximate=True):
        """
        Gaussian Error Linear Unit.
        If approximate=True, uses tanh approximation (as in many Transformers). [web:50]
        """
        x = np.asarray(x)
        if approximate:
            # 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 x^3) )) 
            c = np.sqrt(2 / np.pi)
            return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * np.power(x, 3))))
        else:
            # Exact GELU via erf
            return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))

    @staticmethod
    def mish(x):
        # f(x) = x * tanh(softplus(x)) [web:48]
        return x * np.tanh(Activations.softplus(x))


ACTIVATION_REGISTRY = {
    "linear": Activations.linear,
    "sigmoid": Activations.sigmoid,
    "tanh": Activations.tanh,
    "relu": Activations.relu,
    "leaky_relu": Activations.leaky_relu,
    "elu": Activations.elu,
    "softplus": Activations.softplus,
    "softmax": Activations.softmax,
    "swish": Activations.swish,
    "gelu": Activations.gelu,
    "mish": Activations.mish,
}
