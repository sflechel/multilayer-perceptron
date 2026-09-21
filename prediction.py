import argparse
from numpy.typing import NDArray
import pandas as pd
from pathlib import Path
import logging
import numpy as np

import pickle
from multilayer_perceptron import MultilayerPerceptron
from utils.feature_scaler import FeatureScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def import_from_csv(path: Path) -> pd.DataFrame:
    try:
        dataset: pd.DataFrame = pd.read_csv(path, header=None)
    except Exception as _:
        raise ValueError(f"Failed to parse csv file at {path}")

    if dataset.shape[1] != 32:
        raise ValueError("Dataset must have 32 columns")

    ids: pd.Series = dataset.iloc[:, 0]
    try:
        ids = ids.astype(dtype=int)
    except Exception as _:
        raise ValueError("Ids are not integers")
    if (ids < 0).any():
        raise ValueError("Ids are negative")
    if not ids.is_unique:
        raise ValueError("Ids are not unique")

    features = dataset.iloc[:, 2:]
    features = features.apply(pd.to_numeric, errors="coerce")
    clean_dataset: pd.DataFrame = pd.concat([ids, features], axis=1).dropna()
    if clean_dataset.empty:
        raise ValueError("Dataset is empty after cleaning")

    logging.info(
        f"Dataset successfully imported and cleaned up. {clean_dataset.shape[0]} rows remains out of {dataset.shape[0]}"
    )
    return clean_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="Path to dataset for prediction")
    parser.add_argument(
        "weights", type=str, help="Path to model weights for prediction"
    )

    args = parser.parse_args()
    data: pd.DataFrame = import_from_csv(Path(args.path))

    with open(args.weights, "rb") as model_artifacts_file:
        bundle = pickle.load(model_artifacts_file)

    mlp: MultilayerPerceptron = bundle["mlp"]
    scaler: FeatureScaler = bundle["scaler"]

    print(data.shape)
    X_pred: NDArray[np.float64] = data.drop(0, axis=1).values
    print(X_pred.shape)
    X_pred = scaler.transform(X_pred)

    predictions: NDArray[np.float64] = mlp.forward(X_pred)

    for i, prob in enumerate(predictions, start=0):
        likelihood = prob[0] * 100
        print(
            f"patient {data.iloc[i, 0]} has {likelihood:.2f}% likelyhood of having malignant breast cancer"
        )


if __name__ == "__main__":
    main()
