from pathlib import Path
import logging
from typing import cast
import pandas as pd
import argparse

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

    diagnoses: pd.Series = dataset.iloc[:, 1]
    allowed_values = {"B", "M"}
    if not diagnoses.isin(allowed_values).all():
        raise ValueError("Diagnosis column contain values other than B and M")
    diagnoses = diagnoses.map({"M": 1.0, "B": 0.0})
    print(diagnoses.sum() / len(diagnoses))

    features = dataset.iloc[:, 2:]
    features = features.apply(pd.to_numeric, errors="coerce")
    clean_dataset: pd.DataFrame = pd.concat([ids, diagnoses, features], axis=1).dropna()
    if clean_dataset.empty:
        raise ValueError("Dataset is empty after cleaning")

    logging.info(
        f"Dataset successfully imported and cleaned up. {clean_dataset.shape[0]} rows remains out of {dataset.shape[0]}"
    )
    return clean_dataset


def split_dataset(
    data: pd.DataFrame, training_fraction: float = 0.5, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    shuffled_data = data.sample(frac=1.0, ignore_index=True, random_state=seed)

    split_index: int = int(len(shuffled_data) * training_fraction)
    training_data: pd.DataFrame = cast(pd.DataFrame, shuffled_data[:split_index])
    testing_data: pd.DataFrame = cast(pd.DataFrame, shuffled_data[split_index:])
    return training_data, testing_data


def save_datasets(training_data: pd.DataFrame, testing_data: pd.DataFrame) -> None:
    training_path = Path("output/training_dataset.csv")
    testing_path = Path("output/testing_dataset.csv")
    training_path.parent.mkdir(parents=True, exist_ok=True)
    testing_path.parent.mkdir(parents=True, exist_ok=True)

    training_data.to_csv(training_path, header=False, index=False)
    testing_data.to_csv(testing_path, header=False, index=False)

    print(f"Training dataset saved to {training_path} ({len(training_data)} rows)")
    print(f"Validation dataset saved to {testing_path} ({len(testing_data)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser("Clean and split dataset for training")
    parser.add_argument("path", type=str, help="Path to training data")
    parser.add_argument(
        "--split",
        type=float,
        help="Proportion of data allocated to training",
        default=0.5,
    )
    parser.add_argument(
        "--seed", type=int, help="Random seed for dataset split", default=42
    )
    args = parser.parse_args()
    if args.split <= 0.1 or args.split >= 0.9:
        logging.error("Split parameter is a proportion, must be be between 0.1 and 0.9")
        exit(1)

    try:
        data = import_from_csv(Path(args.path))
    except ValueError as e:
        logging.error(e)
        exit(1)

    training_data: pd.DataFrame
    testing_data: pd.DataFrame
    training_data, testing_data = split_dataset(data, args.split, args.seed)
    save_datasets(training_data, testing_data)


if __name__ == "__main__":
    main()
