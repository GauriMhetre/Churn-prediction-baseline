"""Tests for feature engineering and preprocessing pipeline."""

import numpy as np
import pandas as pd

from src.features import build_preprocessor, split_xy


def test_scaler_mean_on_train() -> None:
    """Test that StandardScaler transforms train data to have mean ≈ 0 and std ≈ 1."""
    # Create simple training data with known statistics
    X_train = pd.DataFrame({
        "tenure": [0, 12, 24, 36, 48],
        "MonthlyCharges": [20.0, 40.0, 60.0, 80.0, 100.0],
        "TotalCharges": [0.0, 480.0, 1440.0, 2880.0, 4800.0],
        "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month", "One year"],
    })

    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    categorical_cols = ["Contract"]

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    # Fit and transform training data
    X_transformed = preprocessor.fit_transform(X_train)

    # Extract the numeric features (first 3 columns before categorical encoding)
    numeric_features = X_transformed[:, :3]

    # Check that transformed numeric features have mean ≈ 0
    assert np.allclose(numeric_features.mean(axis=0), 0.0, atol=1e-10), (
        f"Transformed mean {numeric_features.mean(axis=0)} should be ≈ 0"
    )

    # Check that transformed numeric features have std ≈ 1
    assert np.allclose(numeric_features.std(axis=0), 1.0, atol=1e-10), (
        f"Transformed std {numeric_features.std(axis=0)} should be ≈ 1"
    )


def test_unknown_category_handled() -> None:
    """Test that unseen categories are handled gracefully."""
    # Create training data with specific categories
    X_train = pd.DataFrame({
        "tenure": [0, 12, 24],
        "MonthlyCharges": [20.0, 40.0, 60.0],
        "TotalCharges": [0.0, 480.0, 1440.0],
        "Contract": ["Month-to-month", "One year", "Two year"],
    })

    # Create test data with an unseen category
    X_test = pd.DataFrame({
        "tenure": [36],
        "MonthlyCharges": [80.0],
        "TotalCharges": [2880.0],
        "Contract": ["Unknown-contract-type"],  # Unseen category
    })

    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    categorical_cols = ["Contract"]

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    # Fit on training data and transform test data
    preprocessor.fit(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # Should not raise an error; shape should match expected features
    # After one-hot encoding, we should have 3 numeric + 3 contract categories = 6 features
    assert X_test_transformed.shape[1] == 6, (
        f"Expected 6 features after encoding, got {X_test_transformed.shape[1]}"
    )


def test_feature_names_clean() -> None:
    """Test that customerID and gender are excluded from feature names."""
    # Create data that includes columns to be dropped
    X_train = pd.DataFrame({
        "customerID": ["C1", "C2", "C3"],
        "gender": ["Male", "Female", "Male"],
        "tenure": [0, 12, 24],
        "MonthlyCharges": [20.0, 40.0, 60.0],
        "TotalCharges": [0.0, 480.0, 1440.0],
        "Contract": ["Month-to-month", "One year", "Two year"],
    })

    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    categorical_cols = ["Contract"]

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    preprocessor.fit(X_train)

    # Get feature names after transformation
    feature_names = preprocessor.get_feature_names_out()

    # Check that customerID and gender are not in feature names
    feature_names_str = " ".join(feature_names)
    assert "customerID" not in feature_names_str, (
        "customerID should not appear in feature names"
    )
    assert "gender" not in feature_names_str, (
        "gender should not appear in feature names"
    )

    # Check that numeric and categorical features are present
    assert any("tenure" in name for name in feature_names), (
        "tenure should appear in feature names"
    )
    assert any("Contract" in name for name in feature_names), (
        "Contract should appear in feature names"
    )


def test_split_xy_basic() -> None:
    """Test that split_xy correctly separates features and target."""
    df = pd.DataFrame({
        "customerID": ["C1", "C2", "C3"],
        "gender": ["Male", "Female", "Male"],
        "tenure": [0, 12, 24],
        "Churn": [0, 1, 0],
    })

    cfg = {
        "target": "Churn",
        "id_col": "customerID",
        "drop_cols": ["customerID", "gender"],
    }

    X, y = split_xy(df, cfg)

    # Check that X does not contain dropped columns or target
    assert "customerID" not in X.columns, "customerID should be removed from X"
    assert "gender" not in X.columns, "gender should be removed from X"
    assert "Churn" not in X.columns, "Churn should be removed from X"

    # Check that X contains tenure
    assert "tenure" in X.columns, "tenure should remain in X"

    # Check that y is the target
    assert len(y) == 3, "y should have 3 rows"
    assert (y == df["Churn"]).all(), "y should match target values"
