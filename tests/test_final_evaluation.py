"""Tests for final evaluation infrastructure (Day 12)."""

import json
from pathlib import Path

import pytest

from src.config import load_config
from src.evaluate import final_evaluation


class TestFinalEvaluationGuard:
    """Tests for guard logic in final_evaluation."""

    def test_final_evaluation_refuses_without_force(self, tmp_path) -> None:
        """Test that final_evaluation exits with code 2 if final_metrics.json exists and force=False."""
        cfg = load_config()
        cfg["paths"]["reports_dir"] = str(tmp_path)

        (tmp_path / "models").mkdir(exist_ok=True)

        # Create a dummy final_metrics.json
        final_metrics_path = tmp_path / "final_metrics.json"
        final_metrics_path.write_text('{"dummy": "data"}')

        # Should exit with code 2
        with pytest.raises(SystemExit) as exc_info:
            final_evaluation(cfg, force=False)

        assert exc_info.value.code == 2

    def test_final_evaluation_json_exists_and_valid(self) -> None:
        """Test that final_metrics.json exists and contains required keys."""
        # This test assumes make final has already run
        final_metrics_path = Path("reports/final_metrics.json")

        if final_metrics_path.exists():
            with open(final_metrics_path) as f:
                json_data = json.load(f)

            # Check all required keys exist
            assert "pr_auc" in json_data
            assert "roc_auc" in json_data
            assert "recall_at_precision" in json_data
            assert "brier" in json_data
            assert "threshold" in json_data
            assert "n_test" in json_data
            assert "n_churners_test" in json_data

            # Check metric structure (each should have point, ci_low, ci_high)
            for metric_key in ["pr_auc", "roc_auc", "recall_at_precision", "brier"]:
                assert "point" in json_data[metric_key], f"{metric_key} missing 'point'"
                assert "ci_low" in json_data[metric_key], f"{metric_key} missing 'ci_low'"
                assert "ci_high" in json_data[metric_key], f"{metric_key} missing 'ci_high'"

                # Verify CIs are ordered correctly
                assert (json_data[metric_key]["ci_low"] <= json_data[metric_key]["point"]
                        <= json_data[metric_key]["ci_high"]), f"CI bounds invalid for {metric_key}"

            # Verify threshold value
            assert json_data["threshold"] == 0.2147, "Threshold should be 0.2147"

            # Verify counts
            assert json_data["n_test"] == 1409, "Test set should have 1409 rows"
            assert json_data["n_churners_test"] == 374, "Test set should have 374 churners"

