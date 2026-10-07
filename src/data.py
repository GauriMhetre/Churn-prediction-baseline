"""Data loading and database storage module."""

import argparse
import logging
import sqlite3
from pathlib import Path

import pandas as pd

from src.config import load_config

logger = logging.getLogger(__name__)

EXPECTED_RAW_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]


def load_raw(path: str) -> pd.DataFrame:
    """Load raw CSV data, perform initial type conversions and validations.

    Args:
        path: Path to the raw CSV file.

    Returns:
        pd.DataFrame: Cleaned raw DataFrame with 'churn' integer column.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If expected columns are missing or customerID values are not unique.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        msg = f"Raw CSV data file not found: {path}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    df = pd.read_csv(csv_path)

    missing_cols = set(EXPECTED_RAW_COLUMNS) - set(df.columns)
    if missing_cols:
        msg = f"Raw dataset missing expected columns: {sorted(missing_cols)}"
        logger.error(msg)
        raise ValueError(msg)

    if not df["customerID"].is_unique:
        duplicates = df[df["customerID"].duplicated()]["customerID"].tolist()
        msg = f"Duplicate customerID found in dataset: {duplicates[:5]}"
        logger.error(msg)
        raise ValueError(msg)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)
    df["churn"] = (df["Churn"] == "Yes").astype(int)

    return df


def to_sqlite(df: pd.DataFrame, db_path: str, table: str = "customers") -> None:
    """Save DataFrame to SQLite database and create indices.

    Args:
        df: DataFrame to save.
        db_path: Target SQLite database file path.
        table: Target database table name.
    """
    database_path = Path(db_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)

    db_df = df.drop(columns=["Churn"]) if ("Churn" in df.columns and "churn" in df.columns) else df

    with sqlite3.connect(database_path) as conn:
        db_df.to_sql(table, conn, if_exists="replace", index=False)
        cursor = conn.cursor()
        cursor.execute(
            f"CREATE UNIQUE INDEX IF NOT EXISTS idx_{table}_customerID ON {table}(customerID)"
        )
        cursor.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{table}_Contract ON {table}(Contract)"
        )
        conn.commit()

    logger.info(f"Database build complete: {table} saved to {db_path}")


def main() -> None:
    """CLI entrypoint for data operations."""
    parser = argparse.ArgumentParser(description="Data processing and SQLite CLI")
    parser.add_argument(
        "--build-db",
        action="store_true",
        help="Load raw CSV data and build SQLite database",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    cfg = load_config()

    if args.build_db:
        logger.info("Building SQLite database from raw CSV...")
        raw_df = load_raw(cfg["paths"]["raw_csv"])
        to_sqlite(raw_df, cfg["paths"]["db"])
        print(f"Successfully loaded {len(raw_df)} rows into SQLite table 'customers'.")


if __name__ == "__main__":
    main()
