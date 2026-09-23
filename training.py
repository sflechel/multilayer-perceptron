import argparse
from pathlib import Path
from typing import Dict, List
from numpy.typing import NDArray
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import logging
import json
import os

from multilayer_perceptron import MultilayerPerceptron
from utils.feature_scaler import FeatureScaler
from layer_class import Layer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a multilayer perceptron for WDBC binary classification."
    )
    parser.add_argument(
        "--layer",
        nargs="+",
        type=int,
        default=[24, 24],
        help="Sizes of the hidden layers (e.g., 24 24 24)",
    )
    parser.add_argument(
        "--epochs", type=int, default=70, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", "-bs", type=int, default=8, help="Mini-batch size"
    )
    parser.add_argument(
        "--learning_rate",
        "-lr",
        type=float,
        default=0.01,
        help="Learning rate for gradient descent",
    )
    parser.add_argument(
        "-activ",
        "--activation",
        type=str,
        default="relu",
        choices=["relu", "sigmoid", "tanh", "linear"],
        help="Activation function",
    )
    parser.add_argument(
        "-init",
        "--initializer",
        type=str,
        default="he",
        choices=["he", "xavier", "zero"],
        help="Weight initialization method",
    )
    parser.add_argument(
        "-reg",
        "--regularization",
        type=str,
        default="none",
        choices=["none", "l1", "l2"],
        help="Regularization helps to prevent overfitting",
    )
    parser.add_argument(
        "-reg_lambda",
        "--regularization_lambda",
        type=float,
        default=1,
        help="Parameter for l1 or l2 regularization",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="How long must validation loss plateau before we stop",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for weight initialization"
    )

    return parser.parse_args()


def plot_metrics(history: Dict[str, List[float]], args: argparse.Namespace) -> None:
    _, axes = plt.subplots(2, 2, figsize=(14, 10))
    (ax1, ax2), (ax3, ax4) = axes

    ax1.plot(history["training_loss"], label="Train Loss", color="#1f77b4", linewidth=2)
    ax1.plot(
        history["validation_loss"],
        label="Validation Loss",
        color="#ff7f0e",
        linewidth=2,
        linestyle="--",
    )
    ax1.set_title(f"Loss Curves (Layers: {args.layer})")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Binary Cross-Entropy Loss")
    ax1.legend()
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax2.plot(
        history["training_accuracy"],
        label="Train Accuracy",
        color="#1f77b4",
        linewidth=2,
    )
    ax2.plot(
        history["validation_accuracy"],
        label="Validation Accuracy",
        linewidth=2,
        color="#ff7f0e",
        linestyle="--",
    )
    ax2.set_title(f"Accuracy Curves (Layers: {args.layer})")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, linestyle=":", alpha=0.6)

    ax3.plot(
        history["training_precision"],
        label="Train Precision",
        color="#1f77b4",
        linewidth=2,
    )
    ax3.plot(
        history["validation_precision"],
        label="Validation Precision",
        linewidth=2,
        color="#ff7f0e",
        linestyle="--",
    )
    ax3.set_title(f"Precision Curves (Layers: {args.layer})")
    ax3.set_xlabel("Epochs")
    ax3.set_ylabel("Precision")
    ax3.legend()
    ax3.grid(True, linestyle=":", alpha=0.6)

    ax4.plot(
        history["training_recall"],
        label="Train Recall",
        color="#1f77b4",
        linewidth=2,
    )
    ax4.plot(
        history["validation_recall"],
        label="Validation Recall",
        linewidth=2,
        color="#ff7f0e",
        linestyle="--",
    )
    ax4.set_title(f"Recall Curves (Layers: {args.layer})")
    ax4.set_xlabel("Epochs")
    ax4.set_ylabel("Recall")
    ax4.legend()
    ax4.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.show()


def export_training_history(
    history: dict, args: argparse.Namespace, log_dir: str = "logs"
) -> None:
    os.makedirs(log_dir, exist_ok=True)

    layers_str = "-".join(map(str, args.layer))
    filename = (
        f"init{args.initializer}_layers{layers_str}_"
        f"bs{args.batch_size}_"
        f"lr{args.learning_rate}_reg{args.regularization}_"
        f"lambda{args.regularization_lambda}_"
        f"seed{args.seed}_patience{args.patience}.json"
    )
    filepath = os.path.join(log_dir, filename)

    try:
        with open(filepath, "w") as log_file:
            json.dump(history, log_file, indent=4)
    except Exception as e:
        logging.error(e)
        exit(1)

    print(f"Training metrics history saved to {filepath}")


def main() -> None:
    args: argparse.Namespace = parse_arguments()

    try:
        training_dataset: pd.DataFrame = pd.read_csv(
            Path("output/training_dataset.csv"), header=None
        )
        testing_dataset: pd.DataFrame = pd.read_csv(
            Path("output/testing_dataset.csv"), header=None
        )
    except Exception as e:
        logging.error(e)
        exit(1)

    X_train: NDArray[np.float64] = training_dataset.iloc[:, 2:].values
    y_train: NDArray[np.float64] = training_dataset.iloc[:, 1].values.reshape(-1, 1)
    X_val: NDArray[np.float64] = testing_dataset.iloc[:, 2:].values
    y_val: NDArray[np.float64] = testing_dataset.iloc[:, 1].values.reshape(-1, 1)

    scaler = FeatureScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    mlp = MultilayerPerceptron(
        args.regularization, args.regularization_lambda, args.patience
    )
    nb_features: int = X_train.shape[1]
    prev_dim: int = nb_features

    i: int = 0
    for hidden_dim in args.layer:
        mlp.add(
            Layer(
                n_in=prev_dim,
                n_out=hidden_dim,
                activation=args.activation,
                initializer=args.initializer,
                seed=args.seed + i,
            )
        )
        prev_dim = hidden_dim
        i += 1

    mlp.add(
        Layer(
            n_in=prev_dim,
            n_out=1,
            activation="sigmoid",
            initializer="xavier",
            seed=args.seed + i,
        )
    )

    logging.info(
        f"Starting training for {args.epochs + 1} epochs with batch size {args.batch_size}..."
    )

    history = mlp.fit(
        features=X_train,
        targets=y_train,
        nb_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        features_val=X_val,
        targets_val=y_val,
    )

    try:
        with open("output/model_artifacts.pkl", "wb") as export_file:
            pickle.dump({"mlp": mlp, "scaler": scaler}, export_file)
    except Exception as e:
        logging.error(e)
        exit(1)

    export_training_history(history, args)
    plot_metrics(history, args)


if __name__ == "__main__":
    main()
