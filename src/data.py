"""Data loading and database storage module."""

import argparse
import logging
import sqlite3
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

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

    # Drop 'Churn' if both 'Churn' and 'churn' exist, otherwise use df as-is
    if "Churn" in df.columns and "churn" in df.columns:
        db_df = df.drop(columns=["Churn"])
    else:
        db_df = df

    with sqlite3.connect(database_path) as conn:
        db_df.to_sql(table, conn, if_exists="replace", index=False)
        cursor = conn.cursor()
        idx_customer_id = f"CREATE UNIQUE INDEX IF NOT EXISTS idx_{table}_customerID ON {table}(customerID)"
        cursor.execute(idx_customer_id)
        idx_contract = f"CREATE INDEX IF NOT EXISTS idx_{table}_Contract ON {table}(Contract)"
        cursor.execute(idx_contract)
        conn.commit()

    logger.info(f"Database build complete: {table} saved to {db_path}")


def make_splits(
    df: pd.DataFrame, cfg: dict
) -> dict[str, pd.DataFrame]:
    """Create stratified train/val/test splits based on churn target.

    Two-step process: first split off test, then split remainder into train/val.
    Both splits stratify on churn and use the seed from config.

    Args:
        df: Input DataFrame with 'churn' column.
        cfg: Config dict with 'split' section containing train/val/test fractions
             and 'seed' for random state.

    Returns:
        dict: {"train": train_df, "val": val_df, "test": test_df}
    """
    seed = cfg.get("seed", 42)
    val_frac = cfg["split"]["val"]
    test_frac = cfg["split"]["test"]

    # Step 1: split off test set
    train_val, test = train_test_split(
        df,
        test_size=test_frac,
        stratify=df["churn"],
        random_state=seed,
    )

    # Step 2: split train_val into train and val
    # Adjust val_size relative to train_val size
    val_size_adjusted = val_frac / (1.0 - test_frac)
    train, val = train_test_split(
        train_val,
        test_size=val_size_adjusted,
        stratify=train_val["churn"],
        random_state=seed,
    )

    logger.info(
        f"Splits: train={len(train)}, val={len(val)}, test={len(test)}"
    )
    return {"train": train, "val": val, "test": test}


def save_split_ids(splits: dict[str, pd.DataFrame], splits_dir: str) -> None:
    """Save split customer IDs to CSV files.

    Args:
        splits: dict mapping split name to DataFrame.
        splits_dir: Directory to save the ID files.
    """
    splits_path = Path(splits_dir)
    splits_path.mkdir(parents=True, exist_ok=True)

    id_col = "customerID"
    for split_name, split_df in splits.items():
        out_file = splits_path / f"{split_name}_ids.csv"
        split_df[[id_col]].to_csv(out_file, index=False)
        logger.info(f"Saved {len(split_df)} IDs to {out_file}")


def load_split(
    df: pd.DataFrame, splits_dir: str, name: str
) -> pd.DataFrame:
    """Load a split's customer IDs and return matching rows from input DataFrame.

    Validates that all IDs in the split file exist in df and raises if not.
    Also checks for overlaps with other splits (if they exist).

    Args:
        df: Full DataFrame to filter.
        splits_dir: Directory containing split ID files.
        name: Split name ("train", "val", or "test").

    Returns:
        pd.DataFrame: Filtered DataFrame with rows matching the split IDs.

    Raises:
        FileNotFoundError: If split ID file is missing.
        ValueError: If any ID is missing from df or if overlaps exist.
    """
    id_col = "customerID"
    splits_path = Path(splits_dir)
    split_file = splits_path / f"{name}_ids.csv"

    if not split_file.exists():
        msg = f"Split ID file not found: {split_file}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    split_ids_df = pd.read_csv(split_file)
    split_ids = set(split_ids_df[id_col])

    # Check for missing IDs
    missing_ids = split_ids - set(df[id_col])
    if missing_ids:
        msg = f"Split '{name}' has {len(missing_ids)} IDs not in dataframe"
        logger.error(msg)
        raise ValueError(msg)

    # Check for overlaps with other splits (if they exist)
    other_splits = {"train", "val", "test"} - {name}
    for other_name in other_splits:
        other_file = splits_path / f"{other_name}_ids.csv"
        if other_file.exists():
            other_ids = set(pd.read_csv(other_file)[id_col])
            overlap = split_ids & other_ids
            if overlap:
                msg = (
                    f"Overlap detected between '{name}' "
                    f"and '{other_name}': {len(overlap)} IDs"
                )
                logger.error(msg)
                raise ValueError(msg)

    return df[df[id_col].isin(split_ids)].reset_index(drop=True)


def main() -> None:
    """CLI entrypoint for data operations."""
    parser = argparse.ArgumentParser(description="Data processing and SQLite CLI")
    parser.add_argument(
        "--build-db",
        action="store_true",
        help="Load raw CSV data and build SQLite database",
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="Create stratified train/val/test splits and save IDs",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    cfg = load_config()

    if args.build_db:
        logger.info("Building SQLite database from raw CSV...")
        raw_df = load_raw(cfg["paths"]["raw_csv"])
        to_sqlite(raw_df, cfg["paths"]["db"])
        print(f"Successfully loaded {len(raw_df)} rows into SQLite "
              f"table 'customers'.")

    if args.split:
        logger.info("Creating stratified train/val/test splits...")
        raw_df = load_raw(cfg["paths"]["raw_csv"])
        splits = make_splits(raw_df, cfg)
        splits_dir = cfg["paths"]["splits_dir"]
        save_split_ids(splits, splits_dir)
        print(f"Split IDs saved to {splits_dir}")


if __name__ == "__main__":
    main()
