"""Download the Telco Customer Churn dataset and verify SHA-256 checksum."""

import hashlib
import sys
import urllib.request
from pathlib import Path

DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
)
EXPECTED_SHA256 = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"
OUTPUT_PATH = Path("data/raw/telco_churn.csv")


def main() -> int:
    """Download dataset, verify SHA-256 checksum, save to data/raw/telco_churn.csv.

    Returns:
        int: 0 on success, 1 on failure.
    """
    print(f"Downloading dataset from {DATA_URL}...")
    try:
        req = urllib.request.urlopen(DATA_URL)
        data = req.read()
    except urllib.error.URLError as exc:
        print(f"Error downloading dataset: {exc}", file=sys.stderr)
        return 1

    file_size = len(data)
    actual_sha256 = hashlib.sha256(data).hexdigest()

    print(f"Downloaded size: {file_size} bytes")
    print(f"SHA-256 checksum: {actual_sha256}")

    if actual_sha256 != EXPECTED_SHA256:
        print(
            f"ERROR: SHA-256 checksum mismatch!\n"
            f"Expected: {EXPECTED_SHA256}\n"
            f"Got:      {actual_sha256}",
            file=sys.stderr,
        )
        return 1

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_bytes(data)
    print(f"Dataset saved successfully to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
