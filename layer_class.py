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

        self.d_weights: NDArray[np.float64] = np.zeros([n_in, n_out])
        self.d_biases: NDArray[np.float64] = np.zeros([1, n_out])

        self.input: NDArray[np.float64] = np.array([])
        self.Z: NDArray[np.float64] = np.array([])

    def forward(self, input: NDArray[np.float64]) -> NDArray[np.float64]:
        self.input = input

        self.Z = np.matmul(self.input, self.weights.T) + self.biases
        return self.activation(self.Z)

    def backward(
        self,
        output_gradient: NDArray[np.float64],
        learning_rate: float,
        regularization: str,
        regularization_lambda: float,
        is_combined_gradient: bool = False,
    ) -> NDArray[np.float64]:

        l2_lambda: float = 0.1
        if is_combined_gradient:
            activation_gradient = output_gradient
        else:
            activation_gradient: NDArray[np.float64] = (
                output_gradient * self.activation_derived(self.Z)
            )

        self.d_weights = (
            np.matmul(activation_gradient.T, self.input) / self.input.shape[0]
            + l2_lambda * self.weights
        )
        if regularization == "l2":
            self.d_weights += 2 * regularization_lambda * self.weights
        elif regularization == "l1":
            self.d_weights += regularization_lambda * np.sign(self.weights)

        self.d_biases = (
            np.sum(activation_gradient, axis=0, keepdims=True) / self.input.shape[0]
        )

        input_gradient = np.matmul(activation_gradient, self.weights)

        self.weights -= learning_rate * self.d_weights
        self.biases -= learning_rate * self.d_biases

        return input_gradient
