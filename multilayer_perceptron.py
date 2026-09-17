from numpy.typing import NDArray
from layer_class import Layer
from typing import List
import numpy as np


def binary_cross_entropy(
    predicted: NDArray[np.float64], target: NDArray[np.float64]
) -> float:
    epsilon: float = 1e-15
    predicted = np.clip(predicted, epsilon, 1 - epsilon)
    cross_entropy: float = -np.mean(
        target * np.log(predicted) + (1 - target) * np.log(1 - predicted)
    )
    return cross_entropy


class Multilayer_Perceptron:
    def __init__(self) -> None:
        self.layers: List[Layer] = []

    def add(self, layer: Layer) -> None:
        self.layers.append(layer)

    def forward(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        out: NDArray[np.float64] = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, loss_gradient: NDArray[np.float64], learning_rate: float):
        grad: NDArray[np.float64] = loss_gradient
        for layer in reversed(self.layers):
            grad = layer.backward(grad, learning_rate)

    def fit(
        self,
        features: NDArray[np.float64],
        targets: NDArray[np.float64],
        nb_epochs: int,
        batch_size: int,
        learning_rate: float,
        seed: int,
    ) -> None:
        nb_samples: np.integer = features.shape[0]

        for epoch in range(nb_epochs):
            indices: NDArray[np.integer] = np.arange(nb_samples)
            rng = np.random.default_rng(seed + epoch)
            rng.shuffle(indices)
            features = features[indices]
            targets = targets[indices]

            for i in range(0, nb_samples, batch_size):
                feature_batch = features[i : i + batch_size]
                target_batch = targets[i : i + batch_size]

                predictions: NDArray[np.float64] = self.forward(feature_batch)
                loss_gradient: NDArray[np.float64] = predictions - target_batch
                self.backward(loss_gradient, learning_rate)
