import numpy as np
from numpy.typing import NDArray


def relu(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.maximum(0, x)


def relu_derivative(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.where(x > 0, 1.0, 0.0)


def sigmoid(x: NDArray[np.float64]) -> NDArray[np.float64]:
    x_clipped = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x_clipped))


def sigmoid_derivative(x: NDArray[np.float64]) -> NDArray[np.float64]:
    s = sigmoid(x)
    return s * (1.0 - s)


def tanh(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.tanh(x)


def tanh_derivative(x: NDArray[np.float64]) -> NDArray[np.float64]:
    t = np.tanh(x)
    return 1.0 - t**2


def linear(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return x


def linear_derivative(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.ones_like(x)


def get_activation(name: str):
    activations = {
        "relu": (relu, relu_derivative),
        "sigmoid": (sigmoid, sigmoid_derivative),
        "tanh": (tanh, tanh_derivative),
        "linear": (linear, linear_derivative),
    }
    if name.lower() not in activations:
        raise ValueError(
            f"Unknown activation function: {name}. Choose from {list(activations.keys())}"
        )
    return activations[name.lower()]
