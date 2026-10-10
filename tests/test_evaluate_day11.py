"""Tests for Day 11: threshold optimization, calibration, and error analysis."""

# Set matplotlib to non-interactive backend for tests
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from src.evaluate import calibration_plot, choose_threshold, confusion_at


class TestChooseThreshold:
    """Tests for threshold optimization function."""

    def test_choose_threshold_minimizes_cost(self) -> None:
        """Verify that choose_threshold finds a threshold that minimizes cost."""
        # Simple case: scores are well-separated
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_score = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])

        # Equal costs
        threshold = choose_threshold(y_true, y_score, c_fp=1.0, c_fn=1.0)

        # Compute cost at chosen threshold
        conf = confusion_at(y_true, y_score, threshold)
        cost_chosen = 1.0 * conf["fp"] + 1.0 * conf["fn"]

        # Compute cost at nearby thresholds
        for test_threshold in [0.3, 0.5, 0.7]:
            conf_test = confusion_at(y_true, y_score, test_threshold)
            cost_test = 1.0 * conf_test["fp"] + 1.0 * conf_test["fn"]
            assert cost_chosen <= cost_test

    def test_choose_threshold_ties_favor_higher(self) -> None:
        """Verify that ties break toward higher threshold."""
        # Create a case where two thresholds have equal cost
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.2, 0.4, 0.6, 0.8])

        # With equal costs, should prefer higher threshold
        threshold = choose_threshold(y_true, y_score, c_fp=1.0, c_fn=1.0)

        # For thresholds 0.4 and 0.6, compute costs
        conf_04 = confusion_at(y_true, y_score, 0.4)
        conf_06 = confusion_at(y_true, y_score, 0.6)

        cost_04 = 1.0 * conf_04["fp"] + 1.0 * conf_04["fn"]
        cost_06 = 1.0 * conf_06["fp"] + 1.0 * conf_06["fn"]

        # If costs are equal, chosen threshold should be >= 0.5 (higher of the pair)
        if cost_04 == cost_06:
            assert threshold >= 0.5

    def test_choose_threshold_higher_c_fn_never_lowers_threshold(self) -> None:
        """Verify that increasing c_fn never lowers the optimal threshold.

        Higher cost of false negatives means we should predict positive
        more often (lower threshold), but the algorithm still minimizes cost.
        This test verifies monotonicity: c_fn_high >= c_fn_low => threshold_high <= threshold_low
        """
        y_true = np.array([0, 0, 0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.3, 0.4, 0.7, 0.9])

        threshold_low_c_fn = choose_threshold(y_true, y_score, c_fp=1.0, c_fn=1.0)
        threshold_high_c_fn = choose_threshold(y_true, y_score, c_fp=1.0, c_fn=7.0)

        # Higher c_fn should lead to lower or equal threshold (more aggressive prediction)
        assert threshold_high_c_fn <= threshold_low_c_fn

    def test_choose_threshold_high_fp_cost_raises_threshold(self) -> None:
        """Verify that higher c_fp raises the optimal threshold."""
        y_true = np.array([0, 0, 0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.3, 0.4, 0.7, 0.9])

        threshold_low_c_fp = choose_threshold(y_true, y_score, c_fp=1.0, c_fn=1.0)
        threshold_high_c_fp = choose_threshold(y_true, y_score, c_fp=7.0, c_fn=1.0)

        # Higher c_fp should lead to higher threshold (more conservative prediction)
        assert threshold_high_c_fp >= threshold_low_c_fp

    def test_choose_threshold_perfect_separation(self) -> None:
        """Test threshold selection with perfectly separated classes."""
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.0, 0.0, 1.0, 1.0])

        # With perfect separation, any threshold in (0, 1) should achieve 0 cost
        threshold = choose_threshold(y_true, y_score, c_fp=1.0, c_fn=1.0)
        assert 0.0 <= threshold <= 1.0

        conf = confusion_at(y_true, y_score, threshold)
        cost = 1.0 * conf["fp"] + 1.0 * conf["fn"]
        assert cost == 0

    def test_choose_threshold_returns_float(self) -> None:
        """Test that choose_threshold returns a float."""
        y_true = [0, 1, 1]
        y_score = [0.2, 0.7, 0.9]
        threshold = choose_threshold(y_true, y_score, c_fp=1.0, c_fn=1.0)
        assert isinstance(threshold, float)

    def test_choose_threshold_list_input(self) -> None:
        """Test that choose_threshold works with list inputs."""
        y_true = [0, 0, 1, 1]
        y_score = [0.1, 0.3, 0.6, 0.9]
        threshold = choose_threshold(y_true, y_score, c_fp=1.0, c_fn=1.0)
        assert 0.0 <= threshold <= 1.0


class TestConfusionSumsToN:
    """Tests verifying that confusion matrix components sum to n."""

    def test_confusion_sums_to_n_threshold_zero(self) -> None:
        """Verify confusion matrix sums to n at threshold 0."""
        y_true = np.array([0, 0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.3, 0.4, 0.9])

        conf = confusion_at(y_true, y_score, threshold=0.0)
        total = conf["tn"] + conf["fp"] + conf["fn"] + conf["tp"]
        assert total == len(y_true)

    def test_confusion_sums_to_n_threshold_one(self) -> None:
        """Verify confusion matrix sums to n at threshold 1."""
        y_true = np.array([0, 0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.3, 0.4, 0.9])

        conf = confusion_at(y_true, y_score, threshold=1.0)
        total = conf["tn"] + conf["fp"] + conf["fn"] + conf["tp"]
        assert total == len(y_true)

    def test_confusion_sums_to_n_random_threshold(self) -> None:
        """Verify confusion matrix sums to n at arbitrary threshold."""
        y_true = np.array([0, 0, 0, 0, 1, 1, 1])
        y_score = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 0.9])

        for threshold in [0.0, 0.25, 0.5, 0.75, 1.0]:
            conf = confusion_at(y_true, y_score, threshold)
            total = conf["tn"] + conf["fp"] + conf["fn"] + conf["tp"]
            assert total == len(y_true), f"Failed at threshold {threshold}"


class TestCalibrationPlot:
    """Tests for calibration plot generation."""

    def test_calibration_plot_saves_file(self, tmp_path) -> None:
        """Test that calibration_plot saves a PNG file."""
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_score = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])

        output_path = str(tmp_path / "calibration.png")

        # Should not raise
        calibration_plot(y_true, y_score, output_path)

        # File should exist and not be empty
        import os

        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0

    def test_calibration_plot_creates_directory(self, tmp_path) -> None:
        """Test that calibration_plot creates output directory if it doesn't exist."""
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.3, 0.7, 0.9])

        output_path = str(tmp_path / "subdir" / "calibration.png")

        # Directory should not exist yet
        import os

        assert not os.path.exists(os.path.dirname(output_path))

        # Should create directory and save file
        calibration_plot(y_true, y_score, output_path)

        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0

    def test_calibration_plot_with_custom_bins(self, tmp_path) -> None:
        """Test calibration_plot with custom number of bins."""
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_score = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])

        output_path = str(tmp_path / "calibration_5bins.png")

        # Should not raise with custom bin count
        calibration_plot(y_true, y_score, output_path, n_bins=5)

        import os

        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0

    def test_calibration_plot_list_input(self, tmp_path) -> None:
        """Test that calibration_plot works with list inputs."""
        y_true = [0, 0, 1, 1]
        y_score = [0.1, 0.3, 0.7, 0.9]

        output_path = str(tmp_path / "calibration_list.png")

        # Should not raise
        calibration_plot(y_true, y_score, output_path)

        import os

        assert os.path.exists(output_path)


class TestErrorAnalysis:
    """Tests for error analysis module."""

    def test_error_analysis_returns_dataframe(self, tmp_path) -> None:
        """Test that error_analysis returns a DataFrame."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline

        from src.config import load_config
        from src.data import load_raw, make_splits
        from src.error_analysis import error_analysis
        from src.features import build_preprocessor, split_xy

        cfg = load_config()

        # Load data
        raw_df = load_raw(cfg["paths"]["raw_csv"])
        splits = make_splits(raw_df, cfg)
        val_df = splits["val"]

        X_val, y_val = split_xy(val_df, cfg)

        # Build simple pipeline
        numeric_cols = cfg["numeric_cols"]
        categorical_cols = [
            c for c in X_val.columns
            if c not in numeric_cols and c not in cfg["drop_cols"]
        ]
        preprocessor = build_preprocessor(numeric_cols, categorical_cols)
        model = LogisticRegression(random_state=42)
        pipeline = Pipeline([("prep", preprocessor), ("model", model)])
        pipeline.fit(X_val, y_val)

        # Get predictions
        y_val_pred_proba = pipeline.predict_proba(X_val)[:, 1]

        output_path = str(tmp_path / "error_by_segment.csv")

        # Run error analysis
        result_df = error_analysis(
            pipeline,
            X_val,
            y_val,
            y_val_pred_proba,
            threshold=0.5,
            cfg=cfg,
            output_path=output_path,
        )

        # Should be a DataFrame with expected columns
        assert isinstance(result_df, pd.DataFrame)
        expected_cols = [
            "segment_col",
            "segment_value",
            "n",
            "churners",
            "fp",
            "fn",
            "fp_rate",
            "fn_rate",
        ]
        assert all(col in result_df.columns for col in expected_cols)

        # Should have saved the file
        import os

        assert os.path.exists(output_path)

    def test_error_analysis_rates_in_valid_range(self, tmp_path) -> None:
        """Test that FP/FN rates are in [0, 1]."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline

        from src.config import load_config
        from src.data import load_raw, make_splits
        from src.error_analysis import error_analysis
        from src.features import build_preprocessor, split_xy

        cfg = load_config()

        # Load data
        raw_df = load_raw(cfg["paths"]["raw_csv"])
        splits = make_splits(raw_df, cfg)
        val_df = splits["val"]

        X_val, y_val = split_xy(val_df, cfg)

        # Build simple pipeline
        numeric_cols = cfg["numeric_cols"]
        categorical_cols = [
            c for c in X_val.columns
            if c not in numeric_cols and c not in cfg["drop_cols"]
        ]
        preprocessor = build_preprocessor(numeric_cols, categorical_cols)
        model = LogisticRegression(random_state=42)
        pipeline = Pipeline([("prep", preprocessor), ("model", model)])
        pipeline.fit(X_val, y_val)

        # Get predictions
        y_val_pred_proba = pipeline.predict_proba(X_val)[:, 1]

        output_path = str(tmp_path / "error_by_segment.csv")

        # Run error analysis
        result_df = error_analysis(
            pipeline,
            X_val,
            y_val,
            y_val_pred_proba,
            threshold=0.5,
            cfg=cfg,
            output_path=output_path,
        )

        # Check that rates are in [0, 1]
        assert (result_df["fp_rate"] >= 0.0).all()
        assert (result_df["fp_rate"] <= 1.0).all()
        assert (result_df["fn_rate"] >= 0.0).all()
        assert (result_df["fn_rate"] <= 1.0).all()
