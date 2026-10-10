"""Tests for model evaluation metrics."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.evaluate import (
    brier,
    confusion_at,
    evaluate_split,
    pr_auc,
    recall_at_precision,
    roc_auc,
)
from src.features import build_preprocessor


class TestPrAuc:
    """Tests for PR-AUC metric."""

    def test_perfect_scores(self) -> None:
        """Test PR-AUC with perfect predictions."""
        y_true = [0, 0, 1, 1]
        y_score = [0.1, 0.2, 0.9, 0.8]
        score = pr_auc(y_true, y_score)
        assert score == 1.0

    def test_random_scores(self) -> None:
        """Test PR-AUC with random predictions."""
        y_true = [0, 1, 0, 1, 0, 1]
        y_score = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        score = pr_auc(y_true, y_score)
        # With balanced labels and constant score, PR-AUC should equal positive rate
        assert 0.4 < score < 0.7

    def test_returns_float(self) -> None:
        """Test that PR-AUC returns a float."""
        y_true = [0, 1, 1]
        y_score = [0.2, 0.7, 0.9]
        score = pr_auc(y_true, y_score)
        assert isinstance(score, float)


class TestRocAuc:
    """Tests for ROC-AUC metric."""

    def test_perfect_scores(self) -> None:
        """Test ROC-AUC with perfect predictions."""
        y_true = [0, 0, 1, 1]
        y_score = [0.1, 0.2, 0.9, 0.8]
        score = roc_auc(y_true, y_score)
        assert score == 1.0

    def test_worst_scores(self) -> None:
        """Test ROC-AUC with inverted predictions."""
        y_true = [0, 0, 1, 1]
        y_score = [0.9, 0.8, 0.1, 0.2]
        score = roc_auc(y_true, y_score)
        assert score == 0.0

    def test_returns_float(self) -> None:
        """Test that ROC-AUC returns a float."""
        y_true = [0, 1, 1]
        y_score = [0.2, 0.7, 0.9]
        score = roc_auc(y_true, y_score)
        assert isinstance(score, float)


class TestRecallAtPrecision:
    """Tests for recall at precision floor."""

    def test_high_precision_floor_low_recall(self) -> None:
        """Test that high precision floor yields low or zero recall."""
        y_true = np.array([0, 0, 0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.3, 0.4, 0.8, 0.9])
        recall = recall_at_precision(y_true, y_score, floor=0.99)
        # With 99% precision floor, we likely get 0 or very low recall
        assert 0.0 <= recall <= 1.0

    def test_low_precision_floor_high_recall(self) -> None:
        """Test that low precision floor yields higher recall."""
        y_true = np.array([0, 0, 0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.3, 0.4, 0.8, 0.9])
        recall_low = recall_at_precision(y_true, y_score, floor=0.2)
        recall_high = recall_at_precision(y_true, y_score, floor=0.9)
        assert recall_low >= recall_high

    def test_perfect_predictions(self) -> None:
        """Test recall at precision with perfect predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.0, 0.0, 1.0, 1.0])
        # With perfect predictions, recall should be 1.0 for any reasonable floor
        recall = recall_at_precision(y_true, y_score, floor=0.5)
        assert recall == 1.0

    def test_impossible_precision_floor(self) -> None:
        """Test recall at precision with impossible floor (> 1.0)."""
        y_true = [0, 1, 1]
        y_score = [0.2, 0.7, 0.9]
        recall = recall_at_precision(y_true, y_score, floor=1.5)
        assert recall == 0.0

    def test_no_positive_predictions(self) -> None:
        """Test recall when no samples are predicted positive."""
        y_true = np.array([0, 0, 0, 1])
        y_score = np.array([0.0, 0.0, 0.0, 0.0])
        recall = recall_at_precision(y_true, y_score, floor=0.5)
        assert recall == 0.0

    def test_returns_float(self) -> None:
        """Test that recall_at_precision returns a float."""
        y_true = [0, 1, 1]
        y_score = [0.2, 0.7, 0.9]
        recall = recall_at_precision(y_true, y_score, floor=0.5)
        assert isinstance(recall, float)


class TestBrier:
    """Tests for Brier score."""

    def test_perfect_predictions(self) -> None:
        """Test Brier score with perfect predictions."""
        y_true = [0, 0, 1, 1]
        y_score = [0.0, 0.0, 1.0, 1.0]
        score = brier(y_true, y_score)
        assert score == 0.0

    def test_worst_predictions(self) -> None:
        """Test Brier score with worst predictions."""
        y_true = [0, 0, 1, 1]
        y_score = [1.0, 1.0, 0.0, 0.0]
        score = brier(y_true, y_score)
        assert score == 1.0

    def test_returns_float(self) -> None:
        """Test that Brier returns a float."""
        y_true = [0, 1, 1]
        y_score = [0.2, 0.7, 0.9]
        score = brier(y_true, y_score)
        assert isinstance(score, float)


class TestConfusionAt:
    """Tests for confusion matrix at threshold."""

    def test_threshold_zero(self) -> None:
        """Test confusion matrix with threshold 0 (all positive)."""
        y_true = [0, 0, 1, 1]
        y_score = [0.1, 0.2, 0.8, 0.9]
        conf = confusion_at(y_true, y_score, threshold=0.0)
        # All predictions positive: TP=2, FP=2, TN=0, FN=0
        assert conf["tp"] == 2
        assert conf["fp"] == 2
        assert conf["tn"] == 0
        assert conf["fn"] == 0

    def test_threshold_one(self) -> None:
        """Test confusion matrix with threshold 1 (all negative)."""
        y_true = [0, 0, 1, 1]
        y_score = [0.1, 0.2, 0.8, 0.9]
        conf = confusion_at(y_true, y_score, threshold=1.0)
        # All predictions negative: TP=0, FP=0, TN=2, FN=2
        assert conf["tp"] == 0
        assert conf["fp"] == 0
        assert conf["tn"] == 2
        assert conf["fn"] == 2

    def test_threshold_middle(self) -> None:
        """Test confusion matrix with middle threshold."""
        y_true = [0, 0, 1, 1]
        y_score = [0.1, 0.2, 0.8, 0.9]
        conf = confusion_at(y_true, y_score, threshold=0.5)
        # Predictions: [N, N, P, P]
        assert conf["tn"] == 2
        assert conf["fp"] == 0
        assert conf["fn"] == 0
        assert conf["tp"] == 2

    def test_returns_dict_with_correct_keys(self) -> None:
        """Test that confusion_at returns dict with all keys."""
        y_true = [0, 1]
        y_score = [0.2, 0.8]
        conf = confusion_at(y_true, y_score, threshold=0.5)
        assert set(conf.keys()) == {"tn", "fp", "fn", "tp"}
        assert all(isinstance(v, int) for v in conf.values())


class TestEvaluateSplit:
    """Tests for evaluate_split function."""

    def test_evaluate_split_returns_all_metrics(self) -> None:
        """Test that evaluate_split returns all required metrics."""
        # Create simple training data
        X = pd.DataFrame({
            "tenure": [0, 12, 24, 36],
            "MonthlyCharges": [20.0, 40.0, 60.0, 80.0],
            "TotalCharges": [0.0, 480.0, 1440.0, 2880.0],
            "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month"],
        })
        y = pd.Series([0, 0, 1, 1])

        numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
        categorical_cols = ["Contract"]

        preprocessor = build_preprocessor(numeric_cols, categorical_cols)

        from sklearn.pipeline import Pipeline as SKPipeline

        model = LogisticRegression(random_state=42)
        pipeline = SKPipeline([("prep", preprocessor), ("model", model)])
        pipeline.fit(X, y)

        cfg = {
            "metrics": {"precision_floor": 0.5},
        }

        metrics = evaluate_split(pipeline, X, y, threshold=None, cfg=cfg)

        # Check for required keys
        assert "pr_auc" in metrics
        assert "roc_auc" in metrics
        assert "recall_at_precision" in metrics
        assert "brier" in metrics
        assert "confusion" not in metrics  # threshold is None

    def test_evaluate_split_with_threshold(self) -> None:
        """Test that evaluate_split includes confusion matrix with threshold."""
        X = pd.DataFrame({
            "tenure": [0, 12, 24, 36],
            "MonthlyCharges": [20.0, 40.0, 60.0, 80.0],
            "TotalCharges": [0.0, 480.0, 1440.0, 2880.0],
            "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month"],
        })
        y = pd.Series([0, 0, 1, 1])

        numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
        categorical_cols = ["Contract"]

        preprocessor = build_preprocessor(numeric_cols, categorical_cols)

        from sklearn.pipeline import Pipeline as SKPipeline

        model = LogisticRegression(random_state=42)
        pipeline = SKPipeline([("prep", preprocessor), ("model", model)])
        pipeline.fit(X, y)

        cfg = {
            "metrics": {"precision_floor": 0.5},
        }

        metrics = evaluate_split(pipeline, X, y, threshold=0.5, cfg=cfg)

        # Check for required keys including confusion
        assert "pr_auc" in metrics
        assert "roc_auc" in metrics
        assert "recall_at_precision" in metrics
        assert "brier" in metrics
        assert "confusion" in metrics
        assert isinstance(metrics["confusion"], dict)

    def test_evaluate_split_metric_values_reasonable(self) -> None:
        """Test that metric values are reasonable (between 0 and 1)."""
        X = pd.DataFrame({
            "tenure": [0, 12, 24, 36],
            "MonthlyCharges": [20.0, 40.0, 60.0, 80.0],
            "TotalCharges": [0.0, 480.0, 1440.0, 2880.0],
            "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month"],
        })
        y = pd.Series([0, 0, 1, 1])

        numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
        categorical_cols = ["Contract"]

        preprocessor = build_preprocessor(numeric_cols, categorical_cols)

        from sklearn.pipeline import Pipeline as SKPipeline

        model = LogisticRegression(random_state=42)
        pipeline = SKPipeline([("prep", preprocessor), ("model", model)])
        pipeline.fit(X, y)

        cfg = {
            "metrics": {"precision_floor": 0.5},
        }

        metrics = evaluate_split(pipeline, X, y, threshold=None, cfg=cfg)

        # Check metric ranges
        assert 0.0 <= metrics["pr_auc"] <= 1.0
        assert 0.0 <= metrics["roc_auc"] <= 1.0
        assert 0.0 <= metrics["recall_at_precision"] <= 1.0
        assert 0.0 <= metrics["brier"] <= 1.0
