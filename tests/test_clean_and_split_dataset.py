import pytest
import pandas as pd
from pathlib import Path
from clean_and_split_dataset import import_from_csv, split_dataset


@pytest.fixture
def valid_32col_csv(tmp_path: Path) -> Path:
    """Creates a temporary valid 32-column headerless CSV file."""
    file_path: Path = tmp_path / "valid.csv"

    row_1 = "842302,M," + ",".join(["1.0"] * 30)
    row_2 = "842517,B," + "0.0," + ",".join(["1.0"] * 29)

    file_path.write_text(f"{row_1}\n{row_2}\n")
    return file_path


def test_import_from_csv_success(valid_32col_csv: Path):
    """Test that a clean 32-column dataset imports successfully."""
    df: pd.DataFrame = import_from_csv(valid_32col_csv)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert df.shape[1] == 32


def test_import_wrong_column_count(tmp_path: Path):
    """Test that files with anything other than 32 columns fail."""
    file_path = tmp_path / "bad_cols.csv"
    file_path.write_text("842302,M,1.0,2.0,3.0\n")

    with pytest.raises(ValueError, match="Dataset must have 32 columns"):
        import_from_csv(file_path)


def test_import_non_integer_ids(tmp_path: Path):
    """Test that non-integer IDs trigger an error."""
    file_path = tmp_path / "bad_ids.csv"
    row = "not_an_id,M," + ",".join(["1.0"] * 30)
    file_path.write_text(f"{row}\n")

    with pytest.raises(ValueError, match="Ids are not integers"):
        import_from_csv(file_path)


def test_import_negative_ids(tmp_path: Path):
    """Test that negative IDs trigger an error."""
    file_path = tmp_path / "neg_ids.csv"
    row = "-842302,B," + ",".join(["1.0"] * 30)
    file_path.write_text(f"{row}\n")

    with pytest.raises(ValueError, match="Ids are negative"):
        import_from_csv(file_path)


def test_import_duplicate_ids(tmp_path: Path):
    """Test that duplicate IDs trigger an error."""
    file_path = tmp_path / "dup_ids.csv"
    row = "842302,M," + ",".join(["1.0"] * 30)
    file_path.write_text(f"{row}\n{row}\n")

    with pytest.raises(ValueError, match="Ids are not unique"):
        import_from_csv(file_path)


def test_import_invalid_diagnosis(tmp_path: Path):
    """Test that diagnoses other than 'B' or 'M' trigger an error."""
    file_path = tmp_path / "bad_diag.csv"
    row = "842302,X," + ",".join(["1.0"] * 30)
    file_path.write_text(f"{row}\n")

    with pytest.raises(
        ValueError, match="Diagnosis column contain values other than B and M"
    ):
        import_from_csv(file_path)


def test_split_dataset():
    """Test that the dataset splits correctly based on the fraction."""
    # Create a dummy DataFrame with 10 rows
    dummy_data = pd.DataFrame({"col1": range(10)})

    train, test = split_dataset(dummy_data, training_fraction=0.6, seed=42)

    assert len(train) == 6
    assert len(test) == 4
