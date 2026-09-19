from numpy.typing import NDArray
from layer_class import Layer
from typing import Dict, List
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


class MultilayerPerceptron:
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
        layers = list(reversed(self.layers))
        grad = layers[0].backward(grad, learning_rate, is_combined_gradient=True)

        for layer in layers[1:]:
            grad = layer.backward(grad, learning_rate)

    def fit(
        self,
        features: NDArray[np.float64],
        targets: NDArray[np.float64],
        features_val: NDArray[np.float64],
        targets_val: NDArray[np.float64],
        nb_epochs: int = 100,
        batch_size: int = 16,
        learning_rate: float = 0.01,
        seed: int = 42,
    ) -> Dict[str, List[float]]:
        history: Dict[str, List[float]] = {
            "training_loss": [],
            "validation_loss": [],
            "training_accuracy": [],
            "validation_accuracy": [],
            "weights_norm": [],
        }
        nb_samples: np.integer = features.shape[0]

        for epoch in range(nb_epochs):
            indices: NDArray[np.integer] = np.arange(nb_samples)
            rng = np.random.default_rng(seed + epoch)
            rng.shuffle(indices)
            features = features[indices]
            targets = targets[indices]

            for i in range(0, nb_samples, batch_size):
                feature_batch = features[i : i + batch_size]
                target_batch = targets[i : i + batch_size].reshape(-1, 1)

                predictions: NDArray[np.float64] = self.forward(feature_batch)
                loss_gradient: NDArray[np.float64] = predictions - target_batch
                self.backward(loss_gradient, learning_rate)

            weights_size: float = (
                np.linalg.norm(self.layers[0].weights)
                + np.linalg.norm(self.layers[1].weights)
                + np.linalg.norm(self.layers[2].weights)
            )
            preds = self.forward(features)
            print(
                preds.min(),
                preds.max(),
                preds.mean(),
                ((preds > 0.5) == targets.reshape(-1, 1)).mean(),
            )
            history["weights_norm"].append(weights_size)
            training_predictions: NDArray[np.float64] = self.forward(features)
            training_loss: float = binary_cross_entropy(training_predictions, targets)
            training_accuracy: float = (
                (training_predictions > 0.5) == targets.reshape(-1, 1)
            ).mean()
            history["training_loss"].append(training_loss)
            history["training_accuracy"].append(training_accuracy)
            validation_predictions: NDArray[np.float64] = self.forward(features_val)
            validation_accuracy: float = (
                (validation_predictions > 0.5) == targets_val.reshape(-1, 1)
            ).mean()
            validation_loss: float = binary_cross_entropy(
                validation_predictions, targets_val
            )
            history["validation_loss"].append(validation_loss)
            history["validation_accuracy"].append(validation_accuracy)
            print(
                f"epoch {epoch + 1}/{nb_epochs} - loss: {training_loss} - val_loss: {validation_loss} - accuracy: {training_accuracy} - val_accuracy: {validation_accuracy} - weights_norm: {weights_size}"
            )

        return history
