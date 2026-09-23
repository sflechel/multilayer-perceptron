from numpy.typing import NDArray
from utils.initializers import get_initializer
from utils.activation import get_activation
import numpy as np


class Layer:
    def __init__(
        self, n_in: int, n_out: int, activation: str, initializer: str, seed: int
    ) -> None:
        weight_init = get_initializer(initializer)
        self.activation, self.activation_derived = get_activation(activation)

        self.weights: NDArray[np.float64] = weight_init(n_in, n_out, seed)
        self.biases: NDArray[np.float64] = np.zeros([1, n_out])

        self.d_weights: NDArray[np.float64] = np.zeros([n_in, n_out])
        self.d_biases: NDArray[np.float64] = np.zeros([1, n_out])

        self.a_weights: NDArray[np.float64] = np.zeros([n_in, n_out])
        self.a_biases: NDArray[np.float64] = np.zeros([1, n_out])

        self.s_weights: NDArray[np.float64] = np.zeros([n_in, n_out])
        self.s_biases: NDArray[np.float64] = np.zeros([1, n_out])

        self.input: NDArray[np.float64] = np.array([])
        self.Z: NDArray[np.float64] = np.array([])

    def forward(self, input: NDArray[np.float64]) -> NDArray[np.float64]:
        self.input = input

        self.Z = np.matmul(self.input, self.weights.T) + self.biases
        return self.activation(self.Z)

    def regularize(self, regularization: str, reg_lambda: float) -> None:
        if regularization == "l2":
            self.d_weights += 2 * reg_lambda * self.weights
        elif regularization == "l1":
            self.d_weights += reg_lambda * np.sign(self.weights)

    def update_weights(
        self,
        learning_rate: float,
        optimization: str,
        beta1: float,
        beta2: float,
        t: int,
        epsilon: float = 1e-8,
    ) -> None:
        if optimization == "none":
            self.weights = self.weights - learning_rate * self.d_weights
            self.biases = self.biases - learning_rate * self.d_biases

        if optimization == "momentum":
            self.a_weights = beta1 * self.a_weights + learning_rate * self.d_weights
            self.a_biases = beta1 * self.a_biases + learning_rate * self.d_biases

            self.weights = self.weights - self.a_weights
            self.biases = self.biases - self.a_biases

        if optimization == "rmsprop":
            self.s_weights = beta2 * self.s_weights + (1 - beta2) * (self.d_weights**2)
            self.s_biases = beta2 * self.s_biases + (1 - beta2) * (self.d_biases**2)

            self.weights = (
                self.weights
                - (learning_rate / (self.s_weights**0.5 + epsilon)) * self.d_weights
            )
            self.biases = (
                self.biases
                - (learning_rate / (self.s_biases**0.5 + epsilon)) * self.d_biases
            )

        if optimization == "adam":
            self.a_weights = beta1 * self.a_weights + (1 - beta1) * self.d_weights
            self.a_biases = beta1 * self.a_weights + (1 - beta1) * self.d_biases

            self.s_weights = beta2 * self.s_weights + (1 - beta2) * (self.d_weights**2)
            self.s_biases = beta2 * self.s_biases + (1 - beta2) * (self.d_biases**2)

            a_weights_hat = self.a_weights / (1 - beta1**t)
            a_biases_hat = self.a_biases / (1 - beta1**t)
            s_weights_hat = self.s_weights / (1 - beta2**t)
            s_biases_hat = self.s_biases / (1 - beta2**t)

            self.weights = (
                self.weights
                - (learning_rate / (s_weights_hat**0.5 + epsilon)) * a_weights_hat
            )
            self.biases = (
                self.biases
                - (learning_rate / (s_biases_hat**0.5 + epsilon)) * a_biases_hat
            )

    def backward(
        self,
        output_gradient: NDArray[np.float64],
        learning_rate: float,
        regularization: str,
        regularization_lambda: float,
        optimization: str,
        beta1: float,
        beta2: float,
        t: int,
        is_combined_gradient: bool = False,
    ) -> NDArray[np.float64]:

        if is_combined_gradient:
            activation_gradient = output_gradient
        else:
            activation_gradient: NDArray[np.float64] = (
                output_gradient * self.activation_derived(self.Z)
            )

        self.d_weights = (
            np.matmul(activation_gradient.T, self.input) / self.input.shape[0]
        )
        self.regularize(regularization, regularization_lambda)

        self.d_biases = (
            np.sum(activation_gradient, axis=0, keepdims=True) / self.input.shape[0]
        )

        input_gradient = np.matmul(activation_gradient, self.weights)

        self.update_weights(learning_rate, optimization, beta1, beta2, t)

        return input_gradient
