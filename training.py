import argparse
from pathlib import Path
from typing import Dict, List
import pandas as pd
import matplotlib.pyplot as plt

from multilayer_perceptron import Multilayer_Perceptron
from utils.feature_scaler import Feature_Scaler
from layer_class import Layer


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
    parser.add_argument("--batch_size", type=int, default=8, help="Mini-batch size")
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="Learning rate for gradient descent",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        choices=["relu", "sigmoid", "tanh", "linear"],
        help="Activation function",
    )
    parser.add_argument(
        "--initializer",
        type=str,
        default="he",
        choices=["he", "xavier", "zero"],
        help="Weight initialization method",
    )

    return parser.parse_args()


def plot_loss(history: Dict[str, List[float]], args: argparse.Namespace) -> None:
    plt.figure(figsize=(9, 5))
    plt.plot(history["training_loss"], label="Train Loss", color="#1f77b4", linewidth=2)
    plt.plot(
        history["validation_loss"],
        label="Validation Loss",
        color="#ff7f0e",
        linewidth=2,
        linestyle="--",
    )
    plt.title(f"MLP Training Curves (Layers: {args.layer})")
    plt.xlabel("Epochs")
    plt.ylabel("Binary Cross-Entropy Loss")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.show()


def main() -> None:
    args: argparse.Namespace = parse_arguments()

    try:
        training_dataset: pd.DataFrame = pd.read_csv(
            Path("output/training_dataset.csv"), header=None
        )
        testing_dataset: pd.DataFrame = pd.read_csv(
            Path("output/testing_dataset.csv"), header=None
        )
    except Exception as _:
        raise ValueError(
            "Could not find datasets, please run clean_and_split_dataset.py first"
        )

    X_train = training_dataset.iloc[:, 2:].values
    y_train = training_dataset.iloc[:, 1].values
    X_val = testing_dataset.iloc[:, 2:].values
    y_val = testing_dataset.iloc[:, 1].values

    scaler = Feature_Scaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    mlp = Multilayer_Perceptron()
    nb_features = X_train.shape[1]
    prev_dim = nb_features

    for hidden_dim in args.layer:
        mlp.add(
            Layer(
                n_in=prev_dim,
                n_out=hidden_dim,
                activation=args.activation,
                initializer=args.initializer,
            )
        )
        prev_dim = hidden_dim

    mlp.add(Layer(n_in=prev_dim, n_out=1, activation="sigmoid", initializer="xavier"))

    print(
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

    plot_loss(history, args)


if __name__ == "__main__":
    main()
