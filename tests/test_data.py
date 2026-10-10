"""Unit tests for data loading and schema validation."""

from pathlib import Path

import pandas as pd
import pytest

from src.data import EXPECTED_RAW_COLUMNS, load_raw


def create_synthetic_df(rows: list[dict]) -> pd.DataFrame:
    """Helper function to build a synthetic DataFrame with all expected raw columns."""
    base_row = {col: "No" for col in EXPECTED_RAW_COLUMNS}
    base_row.update(
        {
            "customerID": "0001-TEST",
            "gender": "Female",
            "SeniorCitizen": 0,
            "tenure": 1,
            "MonthlyCharges": 50.0,
            "TotalCharges": "50.0",
            "Churn": "No",
        }
    )
    df_rows = []
    for r in rows:
        merged = base_row.copy()
        merged.update(r)
        df_rows.append(merged)
    return pd.DataFrame(df_rows)


def test_load_raw_schema(tmp_path: Path) -> None:
    """Test that load_raw converts types correctly and adds churn integer column."""
    rows = [
        {
            "customerID": "0001-AAA",
            "TotalCharges": "100.5",
            "Churn": "No",
            "SeniorCitizen": 0,
        },
        {
            "customerID": "0002-BBB",
            "TotalCharges": " ",
            "Churn": "Yes",
            "SeniorCitizen": 1,
        },
    ]
    df_syn = create_synthetic_df(rows)
    csv_file = tmp_path / "synthetic_telco.csv"
    df_syn.to_csv(csv_file, index=False)

    df_loaded = load_raw(str(csv_file))

    assert "churn" in df_loaded.columns
    assert set(df_loaded["churn"].unique()).issubset({0, 1})
    assert df_loaded.loc[df_loaded["customerID"] == "0001-AAA", "churn"].iloc[0] == 0
    assert df_loaded.loc[df_loaded["customerID"] == "0002-BBB", "churn"].iloc[0] == 1

    # Check TotalCharges numeric conversion with coercion (blank -> NaN)
    field = df_loaded.loc[df_loaded["customerID"] == "0002-BBB", "TotalCharges"].iloc[0]
    assert pd.isna(field)
    field = df_loaded.loc[df_loaded["customerID"] == "0001-AAA", "TotalCharges"].iloc[0]
    assert field == 100.5


def test_unique_ids(tmp_path: Path) -> None:
    """Test that duplicate customerID raises ValueError."""
    rows = [
        {"customerID": "0001-DUP", "Churn": "No"},
        {"customerID": "0001-DUP", "Churn": "Yes"},
    ]
    df_syn = create_synthetic_df(rows)
    csv_file = tmp_path / "duplicate_telco.csv"
    df_syn.to_csv(csv_file, index=False)

    with pytest.raises(ValueError, match="Duplicate customerID"):
        load_raw(str(csv_file))
