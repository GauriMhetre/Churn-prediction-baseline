"""Tests for configuration loading and validation."""

import pytest

from src.config import load_config


def test_split_fractions_sum_to_one() -> None:
    """Assert that train, val, and test split fractions sum to 1.0."""
    cfg = load_config("configs/baseline.yaml")
    split_sum = cfg["split"]["train"] + cfg["split"]["val"] + cfg["split"]["test"]
    assert pytest.approx(split_sum) == 1.0


def test_missing_config_raises() -> None:
    """Assert that load_config raises FileNotFoundError for missing file."""
    with pytest.raises(FileNotFoundError):
        load_config("configs/non_existent.yaml")
