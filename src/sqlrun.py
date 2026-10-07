"""SQL runner module to execute profiling and analytics SQL queries on SQLite."""

import logging
import sqlite3
from pathlib import Path

import pandas as pd

from src.config import load_config

logger = logging.getLogger(__name__)


def run_sql_file(db_path: str, sql_path: str) -> pd.DataFrame:
    """Execute a SQL query file on SQLite database and return results as a DataFrame.

    Args:
        db_path: Path to the SQLite database.
        sql_path: Path to the .sql file.

    Returns:
        pd.DataFrame: Query results.

    Raises:
        FileNotFoundError: If db_path or sql_path does not exist.
    """
    database_path = Path(db_path)
    query_file_path = Path(sql_path)

    if not database_path.exists():
        msg = f"SQLite database file not found: {db_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    if not query_file_path.exists():
        msg = f"SQL file not found: {sql_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    query_text = query_file_path.read_text(encoding="utf-8")

    with sqlite3.connect(database_path) as conn:
        df = pd.read_sql_query(query_text, conn)

    return df


def run_all(
    db_path: str = "data/churn.db",
    sql_dir: str = "sql",
    out_dir: str = "reports",
) -> dict[str, pd.DataFrame]:
    """Execute all SQL files in sql_dir and save outputs to reports/sql_<name>.csv.

    Args:
        db_path: Path to SQLite database.
        sql_dir: Directory containing .sql query files.
        out_dir: Directory to save CSV output reports.

    Returns:
        dict[str, pd.DataFrame]: Dictionary mapping report name to DataFrame.
    """
    sql_directory = Path(sql_dir)
    output_directory = Path(out_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    results: dict[str, pd.DataFrame] = {}
    sql_files = sorted(sql_directory.glob("*.sql"))

    for sql_file in sql_files:
        name_stem = sql_file.stem
        # Strip leading numeric prefix e.g. 01_profile -> profile
        query_name = name_stem.split("_", 1)[1] if "_" in name_stem else name_stem
        logger.info(f"Running SQL query: {sql_file.name} -> report name: sql_{query_name}.csv")

        df = run_sql_file(db_path, str(sql_file))
        out_csv_path = output_directory / f"sql_{query_name}.csv"
        df.to_csv(out_csv_path, index=False)
        results[query_name] = df
        logger.info(f"Wrote query output to {out_csv_path}")

    return results


def main() -> None:
    """CLI entrypoint for running SQL query scripts."""
    logging.basicConfig(level=logging.INFO)
    cfg = load_config()
    db_path = cfg["paths"]["db"]
    reports_dir = cfg["paths"]["reports_dir"]

    logger.info("Executing all SQL profiling scripts...")
    run_all(db_path=db_path, sql_dir="sql", out_dir=reports_dir)
    print("SQL execution completed successfully.")


if __name__ == "__main__":
    main()
