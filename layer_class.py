from numpy.typing import NDArray
from utils.initializers import get_initializer
from utils.activation import get_activation
import numpy as np


class Layer:
    def __init__(
        self, n_in: int, n_out: int, activation: str = "relu", initializer: str = "he"
    ) -> None:
        weight_init = get_initializer(initializer)
        self.activation, self.activation_derived = get_activation(activation)

        self.weights: NDArray[np.float64] = weight_init(n_in, n_out)
        self.biases: NDArray[np.float64] = np.zeros([1, n_out])

        self.d_weights: NDArray[np.float64] = np.zeros([n_out, n_in])
        self.d_biases: NDArray[np.float64] = np.zeros([1, n_out])

        self.input: NDArray[np.float64] = np.array([])
        self.Z: NDArray[np.float64] = np.array([])

    def forward(self, input: NDArray[np.float64]) -> NDArray[np.float64]:
        self.input = input

        self.Z = np.matmul(self.input, self.weights) + self.biases
        return self.activation(self.Z)

    def backward(
        self, output_gradient: NDArray[np.float64], learning_rate: float
    ) -> NDArray[np.float64]:

        activation_gradient: NDArray[np.float64] = (
            output_gradient * self.activation_derived(self.Z)
        )
        self.d_weights = np.dot(activation_gradient.T, self.input)
        self.d_biases = np.sum(activation_gradient, axis=0, keepdims=True)

        self.weights -= learning_rate * self.d_weights
        self.biases -= learning_rate * self.d_biases

        # input_gradient shape: (batch, n_out) @ (n_out, n_in) = (batch, n_in)
        input_gradient = np.dot(activation_gradient, self.weights)
        return input_gradient
