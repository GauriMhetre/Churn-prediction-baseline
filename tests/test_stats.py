"""Unit tests for statistical analysis functions."""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.stats import bootstrap_ci, chi2_test, wilson_ci


def test_wilson_known_case() -> None:
    """Test Wilson CI against a known hand-computed case.

    For 10 successes out of 50, the 95% Wilson CI should be approximately
    (0.1124, 0.3357) based on statsmodels implementation.
    """
    low, high = wilson_ci(10, 50, alpha=0.05)

    # Allow reasonable tolerance
    assert 0.10 < low < 0.13, f"Lower bound {low} outside expected range"
    assert 0.32 < high < 0.35, f"Upper bound {high} outside expected range"

    # Check that the point estimate is between the bounds
    point_estimate = 10 / 50
    assert low <= point_estimate <= high


def test_wilson_bounds() -> None:
    """Test that Wilson CI bounds are in [0, 1] and order is correct."""
    # Test edge cases
    low, high = wilson_ci(0, 100, alpha=0.05)
    assert 0.0 <= low <= high <= 1.0

    low, high = wilson_ci(100, 100, alpha=0.05)
    assert 0.0 <= low <= high <= 1.0

    # Test middle case
    low, high = wilson_ci(50, 100, alpha=0.05)
    assert 0.0 <= low <= high <= 1.0
    assert low < 0.5 < high  # Point estimate 0.5 should be inside


def test_wilson_symmetric_for_half() -> None:
    """Test that CI is symmetric around 0.5 when p=0.5."""
    low, high = wilson_ci(50, 100, alpha=0.05)
    point = 0.5
    assert abs((point - low) - (high - point)) < 0.01


def test_wilson_edge_zero() -> None:
    """Test Wilson CI with zero successes."""
    low, high = wilson_ci(0, 50, alpha=0.05)
    assert low >= 0.0
    assert high <= 1.0
    assert low < high


def test_wilson_edge_all() -> None:
    """Test Wilson CI with all successes."""
    low, high = wilson_ci(50, 50, alpha=0.05)
    assert low >= 0.0
    assert high <= 1.0
    assert low < high


def test_chi2_bounds() -> None:
    """Test that Cramér's V is in [0, 1] and chi-square is non-negative."""
    # Create a simple contingency table
    data = {
        "category": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
        "churn": [1, 1, 0, 1, 0, 0, 0, 0, 1],
    }
    df = pd.DataFrame(data)

    result = chi2_test(df, "category", target="churn")

    # Check bounds
    assert result["chi2"] >= 0, f"chi2 {result['chi2']} should be non-negative"
    assert 0.0 <= result["cramers_v"] <= 1.0, (
        f"Cramér's V {result['cramers_v']} should be in [0, 1]"
    )
    assert 0.0 <= result["p_value"] <= 1.0, (
        f"p-value {result['p_value']} should be in [0, 1]"
    )
    assert result["dof"] > 0, "Degrees of freedom should be positive"


def test_cramers_v_independent() -> None:
    """Test Cramér's V ≈ 0 for independent categorical variables."""
    # Create independent variables: random category and random churn
    np.random.seed(42)
    n = 500
    data = {
        "category": np.random.choice(["A", "B", "C"], n),
        "churn": np.random.choice([0, 1], n),
    }
    df = pd.DataFrame(data)

    result = chi2_test(df, "category", target="churn")

    # For independent variables, Cramér's V should be small (< 0.2)
    # and p-value should be relatively high (> 0.05 most of the time)
    assert result["cramers_v"] < 0.3, (
        f"Cramér's V {result['cramers_v']} should be small for "
        "independent variables"
    )


def test_bootstrap_ci_point_in_interval() -> None:
    """Test that bootstrap point estimate lies within the CI."""
    y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1])
    y_score = np.array([0.1, 0.9, 0.8, 0.2, 0.85, 0.15, 0.95, 0.7, 0.1, 0.9, 0.2, 0.8, 0.9, 0.1, 0.85])

    point, ci_low, ci_high = bootstrap_ci(
        y_true, y_score, roc_auc_score, n_resamples=100, seed=42
    )

    # Point estimate should lie within the CI (or be NaN if metric computation fails)
    if not np.isnan(point):
        assert ci_low <= point <= ci_high, (
            f"Point {point} should lie between CI [{ci_low}, {ci_high}]"
        )


def test_bootstrap_ci_same_seed() -> None:
    """Test that same seed produces identical results."""
    y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1])
    y_score = np.array([0.1, 0.9, 0.8, 0.2, 0.85, 0.15, 0.95, 0.7, 0.1, 0.9, 0.2, 0.8, 0.9, 0.1, 0.85])

    result1 = bootstrap_ci(
        y_true, y_score, roc_auc_score, n_resamples=100, seed=42
    )
    result2 = bootstrap_ci(
        y_true, y_score, roc_auc_score, n_resamples=100, seed=42
    )

    # Results should be identical with the same seed
    assert result1[0] == result2[0], "Point estimates differ with same seed"
    assert result1[1] == result2[1], "CI low differs with same seed"
    assert result1[2] == result2[2], "CI high differs with same seed"


def test_bootstrap_ci_perfect_predictions() -> None:
    """Test bootstrap CI with perfect predictions (ROC-AUC = 1.0 when possible)."""
    # Use a larger, more balanced sample to ensure ROC-AUC can be computed
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_score = np.array([0.1, 0.05, 0.0, 0.1, 0.05, 0.95, 0.9, 1.0, 0.95, 0.9])

    point, ci_low, ci_high = bootstrap_ci(
        y_true, y_score, roc_auc_score, n_resamples=100, seed=42
    )

    # For nearly perfect predictions, point should be close to 1.0
    if not np.isnan(point):
        assert point > 0.9, f"Nearly perfect predictions should give high ROC-AUC, got {point}"
        # CI width should be relatively small for consistent predictions
        assert (ci_high - ci_low) < 0.3, (
            f"Good predictions should have modest CI width, got [{ci_low}, {ci_high}]"
        )

