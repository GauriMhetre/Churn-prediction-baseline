"""Integration test for data download and verification."""

import hashlib
from pathlib import Path

import pytest

DATA_PATH = Path("data/raw/telco_churn.csv")
EXPECTED_SHA256 = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"


@pytest.mark.skipif(not DATA_PATH.exists(), reason="Raw CSV data file not present")
def test_downloaded_data_sha256() -> None:
    """Test that the downloaded dataset exists and matches the expected SHA-256."""
    data = DATA_PATH.read_bytes()
    actual_sha256 = hashlib.sha256(data).hexdigest()
    assert actual_sha256 == EXPECTED_SHA256
