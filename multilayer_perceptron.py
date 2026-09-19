from numpy.typing import NDArray
from layer_class import Layer
from typing import Dict, List
import numpy as np


class MultilayerPerceptron:
    def __init__(self, reg: str, reg_lambda: float, patience: int) -> None:
        self.layers: List[Layer] = []
        self.reg = reg
        self.reg_lambda = reg_lambda
        self.patience = patience

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
        grad = layers[0].backward(
            grad, learning_rate, self.reg, self.reg_lambda, is_combined_gradient=True
        )

        for layer in layers[1:]:
            grad = layer.backward(grad, learning_rate, self.reg, self.reg_lambda)

    def binary_cross_entropy(
        self, predicted: NDArray[np.float64], target: NDArray[np.float64]
    ) -> float:
        epsilon: float = 1e-15
        predicted = np.clip(predicted, epsilon, 1 - epsilon)
        cross_entropy: float = -np.mean(
            target * np.log(predicted) + (1 - target) * np.log(1 - predicted)
        )

        if self.reg == "l1":
            sum_abs_weights = sum(
                np.sum(np.abs(layer.weights)) for layer in self.layers
            )
            cross_entropy += self.reg_lambda * sum_abs_weights

        if self.reg == "l2":
            sum_sq_weights = sum(np.sum(layer.weights**2) for layer in self.layers)
            cross_entropy += self.reg_lambda * sum_sq_weights

        return cross_entropy

    def log_metrics(
        self,
        epoch: int,
        nb_epochs: int,
        features: NDArray[np.float64],
        targets: NDArray[np.float64],
        validation_predictions: NDArray[np.float64],
        targets_val: NDArray[np.float64],
        loss_val: float,
        history: Dict[str, List[float]],
    ) -> None:

        weights_size: float = sum(
            np.linalg.norm(layer.weights) for layer in self.layers
        )
        history["weights_norm"].append(weights_size)

        training_predictions = self.forward(features)
        training_loss: float = self.binary_cross_entropy(training_predictions, targets)
        training_accuracy: float = ((training_predictions > 0.5) == targets).mean()

        history["training_loss"].append(training_loss)
        history["training_accuracy"].append(training_accuracy)

        validation_accuracy: float = (
            (validation_predictions > 0.5) == targets_val
        ).mean()
        history["validation_loss"].append(loss_val)
        history["validation_accuracy"].append(validation_accuracy)

        print(
            f"epoch {epoch + 1}/{nb_epochs} - loss: {training_loss} - val_loss: {loss_val} "
            f"- accuracy: {training_accuracy} - val_accuracy: {validation_accuracy} - weights_norm: {weights_size}"
        )

    def snapshot(self):
        return (
            [layer.weights.copy() for layer in self.layers],
            [layer.biases.copy() for layer in self.layers],
        )

    def restore(self, weights, biases):
        for layer, w, b in zip(self.layers, weights, biases):
            layer.weights = w
            layer.biases = b

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

        min_delta: float = 1e-4
        best_loss_val: float = float("inf")
        patience_counter: int = 0
        best_weights, best_biases = self.snapshot()

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

            predictions_val = self.forward(features_val)
            loss_val = self.binary_cross_entropy(predictions_val, targets_val)
            self.log_metrics(
                epoch,
                nb_epochs,
                features,
                targets,
                predictions_val,
                targets_val,
                loss_val,
                history,
            )
            if loss_val < best_loss_val - min_delta:
                best_loss_val = loss_val
                patience_counter = 0

                best_weights = [layer.weights.copy() for layer in self.layers]
                best_biases = [layer.biases.copy() for layer in self.layers]
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    break

        self.restore(best_weights, best_biases)
        return history
