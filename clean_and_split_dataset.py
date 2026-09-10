import sys
from pathlib import Path
import logging
import pandas as pd

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

    features = dataset.iloc[:, 2:]
    features = features.apply(pd.to_numeric, errors="coerce")
    clean_dataset: pd.DataFrame = pd.concat([ids, diagnoses, features], axis=1).dropna()
    if clean_dataset.empty:
        raise ValueError("Dataset is empty after cleaning")

    logging.info(
        f"Dataset successfully imported and cleaned up. {clean_dataset.shape[0]} rows remains out of {dataset.shape[0]}"
    )
    return clean_dataset


def main() -> None:
    if len(sys.argv) == 2:
        dataset_path = Path(sys.argv[1])
    else:
        logging.error("Pass in dataset as first and only argument")
        exit(1)
    data = import_from_csv(dataset_path)


if __name__ == "__main__":
    main()
