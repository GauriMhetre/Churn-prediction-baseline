"""Tests for model training and cross-validation."""

import tempfile
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline

from src.features import build_preprocessor
from src.train import (
    cross_validate_model,
    get_models,
    log_run,
)


class TestGetModels:
    """Tests for get_models function."""

    def test_get_models_returns_dict(self) -> None:
        """Test that get_models returns a dictionary."""
        cfg = {
            "seed": 42,
            "models": {
                "logreg": {"C": 1.0, "max_iter": 1000, "class_weight": "balanced"},
                "hgb": {
                    "max_depth": 4,
                    "learning_rate": 0.1,
                    "max_iter": 200,
                    "class_weight": "balanced",
                },
            },
        }
        models = get_models(cfg)
        assert isinstance(models, dict)

    def test_get_models_has_required_keys(self) -> None:
        """Test that get_models returns all required model keys."""
        cfg = {
            "seed": 42,
            "models": {
                "logreg": {"C": 1.0, "max_iter": 1000, "class_weight": "balanced"},
                "hgb": {
                    "max_depth": 4,
                    "learning_rate": 0.1,
                    "max_iter": 200,
                    "class_weight": "balanced",
                },
            },
        }
        models = get_models(cfg)
        assert "dummy" in models
        assert "logreg" in models
        assert "hgb" in models

    def test_get_models_dummy_classifier(self) -> None:
        """Test that dummy model is DummyClassifier with prior strategy."""
        from sklearn.dummy import DummyClassifier

        cfg = {
            "seed": 42,
            "models": {
                "logreg": {"C": 1.0, "max_iter": 1000, "class_weight": "balanced"},
                "hgb": {
                    "max_depth": 4,
                    "learning_rate": 0.1,
                    "max_iter": 200,
                    "class_weight": "balanced",
                },
            },
        }
        models = get_models(cfg)
        assert isinstance(models["dummy"], DummyClassifier)
        assert models["dummy"].strategy == "prior"

    def test_get_models_logreg(self) -> None:
        """Test that logreg model is LogisticRegression with correct params."""
        from sklearn.linear_model import LogisticRegression

        cfg = {
            "seed": 42,
            "models": {
                "logreg": {"C": 1.0, "max_iter": 1000, "class_weight": "balanced"},
                "hgb": {
                    "max_depth": 4,
                    "learning_rate": 0.1,
                    "max_iter": 200,
                    "class_weight": "balanced",
                },
            },
        }
        models = get_models(cfg)
        assert isinstance(models["logreg"], LogisticRegression)
        assert models["logreg"].C == 1.0
        assert models["logreg"].max_iter == 1000

    def test_get_models_hgb(self) -> None:
        """Test that hgb model is HistGradientBoostingClassifier with correct params."""
        from sklearn.ensemble import HistGradientBoostingClassifier

        cfg = {
            "seed": 42,
            "models": {
                "logreg": {"C": 1.0, "max_iter": 1000, "class_weight": "balanced"},
                "hgb": {
                    "max_depth": 4,
                    "learning_rate": 0.1,
                    "max_iter": 200,
                    "class_weight": "balanced",
                },
            },
        }
        models = get_models(cfg)
        assert isinstance(models["hgb"], HistGradientBoostingClassifier)
        assert models["hgb"].max_depth == 4
        assert models["hgb"].learning_rate == 0.1

    def test_get_models_has_random_state(self) -> None:
        """Test that models have random_state set."""
        cfg = {
            "seed": 42,
            "models": {
                "logreg": {"C": 1.0, "max_iter": 1000, "class_weight": "balanced"},
                "hgb": {
                    "max_depth": 4,
                    "learning_rate": 0.1,
                    "max_iter": 200,
                    "class_weight": "balanced",
                },
            },
        }
        models = get_models(cfg)
        assert models["dummy"].random_state == 42
        assert models["logreg"].random_state == 42
        assert models["hgb"].random_state == 42


class TestCrossValidateModel:
    """Tests for cross_validate_model function."""

    def test_cross_validate_returns_dataframe(self) -> None:
        """Test that cross_validate_model returns a DataFrame."""
        X = pd.DataFrame({
            "tenure": [0, 12, 24, 36, 48, 60, 72, 84, 96, 108],
            "MonthlyCharges": [20.0, 40.0, 60.0, 80.0, 100.0, 20.0, 40.0, 60.0, 80.0, 100.0],
            "TotalCharges": [0.0, 480.0, 1440.0, 2880.0, 4800.0, 0.0, 480.0, 1440.0, 2880.0, 4800.0],
            "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month", "One year",
                         "Month-to-month", "One year", "Two year", "Month-to-month", "One year"],
        })
        y = pd.Series([0, 0, 1, 1, 0, 0, 1, 1, 0, 0])

        numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
        categorical_cols = ["Contract"]

        preprocessor = build_preprocessor(numeric_cols, categorical_cols)

        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(random_state=42)
        pipeline = Pipeline([("prep", preprocessor), ("model", model)])

        cfg = {
            "seed": 42,
            "cv": {"n_splits": 3},
        }

        result = cross_validate_model("test_model", pipeline, X, y, cfg)
        assert isinstance(result, pd.DataFrame)

    def test_cross_validate_has_required_columns(self) -> None:
        """Test that cross-validation results have required columns."""
        X = pd.DataFrame({
            "tenure": [0, 12, 24, 36, 48, 60, 72, 84, 96, 108],
            "MonthlyCharges": [20.0, 40.0, 60.0, 80.0, 100.0, 20.0, 40.0, 60.0, 80.0, 100.0],
            "TotalCharges": [0.0, 480.0, 1440.0, 2880.0, 4800.0, 0.0, 480.0, 1440.0, 2880.0, 4800.0],
            "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month", "One year",
                         "Month-to-month", "One year", "Two year", "Month-to-month", "One year"],
        })
        y = pd.Series([0, 0, 1, 1, 0, 0, 1, 1, 0, 0])

        numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
        categorical_cols = ["Contract"]

        preprocessor = build_preprocessor(numeric_cols, categorical_cols)

        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(random_state=42)
        pipeline = Pipeline([("prep", preprocessor), ("model", model)])

        cfg = {
            "seed": 42,
            "cv": {"n_splits": 3},
        }

        result = cross_validate_model("test_model", pipeline, X, y, cfg)
        assert "fold" in result.columns
        assert "pr_auc" in result.columns
        assert "roc_auc" in result.columns

    def test_cross_validate_correct_number_of_folds(self) -> None:
        """Test that cross-validation returns correct number of rows."""
        X = pd.DataFrame({
            "tenure": [0, 12, 24, 36, 48, 60, 72, 84, 96, 108],
            "MonthlyCharges": [20.0, 40.0, 60.0, 80.0, 100.0, 20.0, 40.0, 60.0, 80.0, 100.0],
            "TotalCharges": [0.0, 480.0, 1440.0, 2880.0, 4800.0, 0.0, 480.0, 1440.0, 2880.0, 4800.0],
            "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month", "One year",
                         "Month-to-month", "One year", "Two year", "Month-to-month", "One year"],
        })
        y = pd.Series([0, 0, 1, 1, 0, 0, 1, 1, 0, 0])

        numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
        categorical_cols = ["Contract"]

        preprocessor = build_preprocessor(numeric_cols, categorical_cols)

        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(random_state=42)
        pipeline = Pipeline([("prep", preprocessor), ("model", model)])

        cfg = {
            "seed": 42,
            "cv": {"n_splits": 5},
        }

        result = cross_validate_model("test_model", pipeline, X, y, cfg)
        assert len(result) == 5

    def test_cross_validate_metrics_in_valid_range(self) -> None:
        """Test that cross-validation metrics are in valid range."""
        X = pd.DataFrame({
            "tenure": [0, 12, 24, 36, 48, 60, 72, 84, 96, 108],
            "MonthlyCharges": [20.0, 40.0, 60.0, 80.0, 100.0, 20.0, 40.0, 60.0, 80.0, 100.0],
            "TotalCharges": [0.0, 480.0, 1440.0, 2880.0, 4800.0, 0.0, 480.0, 1440.0, 2880.0, 4800.0],
            "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month", "One year",
                         "Month-to-month", "One year", "Two year", "Month-to-month", "One year"],
        })
        y = pd.Series([0, 0, 1, 1, 0, 0, 1, 1, 0, 0])

        numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
        categorical_cols = ["Contract"]

        preprocessor = build_preprocessor(numeric_cols, categorical_cols)

        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(random_state=42)
        pipeline = Pipeline([("prep", preprocessor), ("model", model)])

        cfg = {
            "seed": 42,
            "cv": {"n_splits": 3},
        }

        result = cross_validate_model("test_model", pipeline, X, y, cfg)
        assert (result["pr_auc"] >= 0.0).all() and (result["pr_auc"] <= 1.0).all()
        assert (result["roc_auc"] >= 0.0).all() and (result["roc_auc"] <= 1.0).all()


class TestLogRun:
    """Tests for log_run function."""

    def test_log_run_creates_file(self) -> None:
        """Test that log_run creates the CSV file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_path = Path(tmpdir) / "runs.csv"

            run_record = {
                "model": "logreg",
                "params": {"C": 1.0, "max_iter": 1000},
                "mean_pr_auc": 0.85,
                "mean_roc_auc": 0.82,
                "std_pr_auc": 0.05,
                "std_roc_auc": 0.03,
            }

            log_run(run_record, str(run_path))

            assert run_path.exists()

    def test_log_run_writes_headers(self) -> None:
        """Test that log_run writes column headers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_path = Path(tmpdir) / "runs.csv"

            run_record = {
                "model": "logreg",
                "params": {"C": 1.0, "max_iter": 1000},
                "mean_pr_auc": 0.85,
                "mean_roc_auc": 0.82,
                "std_pr_auc": 0.05,
                "std_roc_auc": 0.03,
            }

            log_run(run_record, str(run_path))

            df = pd.read_csv(run_path)
            expected_cols = {
                "timestamp",
                "model",
                "params",
                "mean_pr_auc",
                "mean_roc_auc",
                "std_pr_auc",
                "std_roc_auc",
            }
            assert expected_cols.issubset(set(df.columns))

    def test_log_run_appends_multiple_records(self) -> None:
        """Test that log_run appends multiple records."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_path = Path(tmpdir) / "runs.csv"

            run1 = {
                "model": "logreg",
                "params": {"C": 1.0, "max_iter": 1000},
                "mean_pr_auc": 0.85,
                "mean_roc_auc": 0.82,
                "std_pr_auc": 0.05,
                "std_roc_auc": 0.03,
            }
            run2 = {
                "model": "hgb",
                "params": {"max_depth": 4, "learning_rate": 0.1},
                "mean_pr_auc": 0.87,
                "mean_roc_auc": 0.84,
                "std_pr_auc": 0.04,
                "std_roc_auc": 0.02,
            }

            log_run(run1, str(run_path))
            log_run(run2, str(run_path))

            df = pd.read_csv(run_path)
            assert len(df) == 2
            assert df.loc[0, "model"] == "logreg"
            assert df.loc[1, "model"] == "hgb"

    def test_log_run_preserves_metrics(self) -> None:
        """Test that log_run preserves metric values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_path = Path(tmpdir) / "runs.csv"

            run_record = {
                "model": "logreg",
                "params": {"C": 1.0},
                "mean_pr_auc": 0.85,
                "mean_roc_auc": 0.82,
                "std_pr_auc": 0.05,
                "std_roc_auc": 0.03,
            }

            log_run(run_record, str(run_path))

            df = pd.read_csv(run_path)
            assert df.loc[0, "mean_pr_auc"] == 0.85
            assert df.loc[0, "mean_roc_auc"] == 0.82
            assert df.loc[0, "std_pr_auc"] == 0.05
            assert df.loc[0, "std_roc_auc"] == 0.03
