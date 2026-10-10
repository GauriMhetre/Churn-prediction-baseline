"""Unit tests for train/val/test splitting."""

from pathlib import Path

import pandas as pd
import pytest

from src.data import (
    load_split,
    make_splits,
    save_split_ids,
)


def create_test_data(n_rows: int = 100, seed: int = 42) -> pd.DataFrame:
    """Create synthetic DataFrame with customer IDs and churn target."""
    import numpy as np

    np.random.seed(seed)
    # Create roughly 30% churn
    churners = int(n_rows * 0.3)
    non_churners = n_rows - churners
    churn_values = [1] * churners + [0] * non_churners
    np.random.shuffle(churn_values)

    return pd.DataFrame(
        {
            "customerID": [f"CUST{i:05d}" for i in range(n_rows)],
            "churn": churn_values,
            "feature_1": np.random.randn(n_rows),
        }
    )


def test_no_overlap(tmp_path: Path) -> None:
    """Test that train/val/test split IDs are pairwise disjoint."""
    df = create_test_data(1000)
    cfg = {
        "seed": 42,
        "split": {"train": 0.60, "val": 0.20, "test": 0.20},
    }

    splits = make_splits(df, cfg)

    # Check disjoint
    train_ids = set(splits["train"]["customerID"])
    val_ids = set(splits["val"]["customerID"])
    test_ids = set(splits["test"]["customerID"])

    assert len(train_ids & val_ids) == 0
    assert len(train_ids & test_ids) == 0
    assert len(val_ids & test_ids) == 0


def test_covers_all_rows(tmp_path: Path) -> None:
    """Test that union of splits covers all rows."""
    df = create_test_data(1000)
    cfg = {
        "seed": 42,
        "split": {"train": 0.60, "val": 0.20, "test": 0.20},
    }

    splits = make_splits(df, cfg)

    train_ids = set(splits["train"]["customerID"])
    val_ids = set(splits["val"]["customerID"])
    test_ids = set(splits["test"]["customerID"])
    all_split_ids = train_ids | val_ids | test_ids

    assert all_split_ids == set(df["customerID"])


def test_stratification() -> None:
    """Test that stratification keeps churn rate within ±1.5 percentage points."""
    df = create_test_data(1000)
    cfg = {
        "seed": 42,
        "split": {"train": 0.60, "val": 0.20, "test": 0.20},
    }

    splits = make_splits(df, cfg)
    overall_rate = df["churn"].mean()

    for split_name, split_df in splits.items():
        split_rate = split_df["churn"].mean()
        diff = abs(split_rate - overall_rate)
        assert diff <= 0.015, (
            f"{split_name} churn rate {split_rate:.4f} "
            f"differs from overall {overall_rate:.4f} by {diff:.4f}"
        )


def test_deterministic() -> None:
    """Test that same seed produces identical splits, different seed produces different."""
    df = create_test_data(1000)
    cfg1 = {
        "seed": 42,
        "split": {"train": 0.60, "val": 0.20, "test": 0.20},
    }
    cfg2 = {
        "seed": 43,
        "split": {"train": 0.60, "val": 0.20, "test": 0.20},
    }

    splits_1 = make_splits(df, cfg1)
    splits_2_same = make_splits(df, cfg1)
    splits_2_diff = make_splits(df, cfg2)

    # Same seed should give identical splits
    for split_name in ["train", "val", "test"]:
        assert (
            set(splits_1[split_name]["customerID"])
            == set(splits_2_same[split_name]["customerID"])
        )

    # Different seed should give different train IDs (with very high probability)
    assert (
        set(splits_1["train"]["customerID"])
        != set(splits_2_diff["train"]["customerID"])
    )


def test_load_split_detects_overlap(tmp_path: Path) -> None:
    """Test that load_split detects and raises on overlapping split files."""
    df = create_test_data(100)
    cfg = {
        "seed": 42,
        "split": {"train": 0.60, "val": 0.20, "test": 0.20},
    }

    splits = make_splits(df, cfg)
    splits_dir = tmp_path / "splits"

    # Save valid splits
    save_split_ids(splits, str(splits_dir))

    # Verify load works for valid splits
    loaded_train = load_split(df, str(splits_dir), "train")
    assert len(loaded_train) == len(splits["train"])

    # Tamper with val_ids.csv: add a train ID to it
    val_file = splits_dir / "val_ids.csv"
    val_ids_df = pd.read_csv(val_file)
    train_ids_df = pd.read_csv(splits_dir / "train_ids.csv")
    val_ids_df = pd.concat(
        [val_ids_df, train_ids_df.iloc[[0]]], ignore_index=True
    )
    val_ids_df.to_csv(val_file, index=False)

    # Now loading train should detect overlap with val
    with pytest.raises(ValueError, match="Overlap detected"):
        load_split(df, str(splits_dir), "train")
