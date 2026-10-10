"""Model evaluation metrics and reporting module."""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def pr_auc(y_true: np.ndarray | list, y_score: np.ndarray | list) -> float:
    """Compute Precision-Recall AUC using average precision score.

    Args:
        y_true: True binary labels (0 or 1).
        y_score: Target scores or predicted probabilities.

    Returns:
        float: PR-AUC score (0 to 1).
    """
    return float(average_precision_score(y_true, y_score))


def roc_auc(y_true: np.ndarray | list, y_score: np.ndarray | list) -> float:
    """Compute ROC AUC using sklearn.metrics.

    Args:
        y_true: True binary labels (0 or 1).
        y_score: Target scores or predicted probabilities.

    Returns:
        float: ROC AUC score (0 to 1).
    """
    return float(roc_auc_score(y_true, y_score))


def recall_at_precision(
    y_true: np.ndarray | list,
    y_score: np.ndarray | list,
    floor: float,
) -> float:
    """Compute maximum recall where precision >= floor.

    Scans unique score thresholds; for each threshold, computes precision and
    recall. Returns the maximum recall where precision >= floor, or 0.0 if no
    threshold satisfies the constraint.

    Args:
        y_true: True binary labels (0 or 1).
        y_score: Target scores or predicted probabilities (0 to 1).
        floor: Minimum precision threshold (0 to 1).

    Returns:
        float: Maximum recall where precision >= floor; 0.0 if none.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    # Get unique thresholds (including boundaries)
    thresholds = np.unique(np.concatenate([[0.0], y_score, [1.0]]))

    max_recall = 0.0

    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)

        # Count TP and FP
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))

        # Avoid division by zero
        if tp + fp == 0:
            precision = 1.0
        else:
            precision = tp / (tp + fp)

        if precision >= floor:
            recall = tp / np.sum(y_true == 1) if np.sum(y_true == 1) > 0 else 0.0
            max_recall = max(max_recall, recall)

    return max_recall


def brier(y_true: np.ndarray | list, y_score: np.ndarray | list) -> float:
    """Compute Brier score (mean squared error between predicted and true labels).

    Args:
        y_true: True binary labels (0 or 1).
        y_score: Predicted probabilities (0 to 1).

    Returns:
        float: Brier score.
    """
    return float(brier_score_loss(y_true, y_score))


def confusion_at(
    y_true: np.ndarray | list,
    y_score: np.ndarray | list,
    threshold: float,
) -> dict[str, int]:
    """Compute confusion matrix at a specific threshold.

    Args:
        y_true: True binary labels (0 or 1).
        y_score: Target scores or predicted probabilities.
        threshold: Decision threshold for positive prediction.

    Returns:
        dict: Keys tn, fp, fn, tp with integer values.
    """
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def choose_threshold(
    y_true: np.ndarray | list,
    y_score: np.ndarray | list,
    c_fp: float,
    c_fn: float,
) -> float:
    """Find optimal threshold that minimizes misclassification cost.

    Scans all unique thresholds in y_score (plus boundaries 0.0 and 1.0).
    For each threshold, computes cost = c_fp * FP + c_fn * FN.
    Returns the threshold with minimum cost; ties break toward higher threshold
    (conservative prediction).

    Args:
        y_true: True binary labels (0 or 1).
        y_score: Predicted probabilities (0 to 1).
        c_fp: Cost of false positive.
        c_fn: Cost of false negative.

    Returns:
        float: Optimal threshold in [0.0, 1.0].
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    # Get unique thresholds (including boundaries)
    thresholds = np.unique(np.concatenate([[0.0], y_score, [1.0]]))

    best_threshold = 0.5
    best_cost = float("inf")

    # Scan all thresholds; ties favor higher threshold
    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)

        # Compute confusion matrix
        _, fp, fn, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        # Compute cost
        cost = c_fp * fp + c_fn * fn

        # Update if strictly better, or equal but higher threshold
        if cost < best_cost or (cost == best_cost and threshold > best_threshold):
            best_cost = cost
            best_threshold = threshold

    return float(best_threshold)


def calibration_plot(
    y_true: np.ndarray | list,
    y_score: np.ndarray | list,
    path: str,
    n_bins: int = 10,
) -> None:
    """Generate and save a calibration (reliability) diagram.

    Divides predicted probabilities into n_bins equal-width bins.
    For each bin, plots mean predicted probability (x-axis) vs.
    empirical frequency of positives (y-axis). Perfect calibration
    lies on the diagonal y=x.

    Args:
        y_true: True binary labels (0 or 1).
        y_score: Predicted probabilities (0 to 1).
        path: Path to save the PNG file.
        n_bins: Number of bins for calibration plot (default 10).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    # Create bins: [0, 1/n_bins), [1/n_bins, 2/n_bins), ..., [(n_bins-1)/n_bins, 1]
    bin_edges = np.linspace(0, 1, n_bins + 1)

    mean_predicted_probs = []
    empirical_freqs = []
    bin_counts = []

    for i in range(n_bins):
        mask = (y_score >= bin_edges[i]) & (y_score < bin_edges[i + 1])
        # Last bin includes 1.0
        if i == n_bins - 1:
            mask = (y_score >= bin_edges[i]) & (y_score <= bin_edges[i + 1])

        if np.sum(mask) > 0:
            mean_pred = np.mean(y_score[mask])
            empirical_freq = np.mean(y_true[mask])
            mean_predicted_probs.append(mean_pred)
            empirical_freqs.append(empirical_freq)
            bin_counts.append(np.sum(mask))

    # Create figure with calibration curve and histogram
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot 1: Calibration curve
    ax1.plot([0, 1], [0, 1], "k--", label="Perfect calibration", linewidth=1)
    ax1.scatter(
        mean_predicted_probs,
        empirical_freqs,
        s=np.array(bin_counts) * 2,  # Scale point size by bin count
        alpha=0.6,
        label="Observed",
    )
    ax1.set_xlabel("Mean predicted probability")
    ax1.set_ylabel("Empirical frequency of positives")
    ax1.set_title("Calibration Plot (Reliability Diagram)")
    ax1.set_xlim([-0.05, 1.05])
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Plot 2: Histogram of predicted probabilities
    ax2.hist(y_score, bins=30, alpha=0.7, edgecolor="black")
    ax2.set_xlabel("Predicted probability")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Distribution of Predicted Probabilities")
    ax2.grid(alpha=0.3, axis="y")

    plt.tight_layout()

    # Ensure output directory exists
    output_dir = Path(path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(path, dpi=100, bbox_inches="tight")
    logger.info(f"Saved calibration plot to {path}")
    plt.close()


def final_evaluation(cfg: dict, force: bool = False) -> dict[str, Any]:
    """Run final evaluation on test split (Day 12 only).

    Refits best model on train+val, evaluates on test with bootstrap CIs.
    Refuses if reports/final_metrics.json already exists unless force=True.

    Args:
        cfg: Config dict.
        force: If True, overwrite existing final_metrics.json.

    Returns:
        dict: Final metrics with bootstrap CIs.

    Raises:
        SystemExit(2): If final_metrics.json exists and force=False.
    """
    import sys


    final_metrics_path = Path(cfg["paths"]["reports_dir"]) / "final_metrics.json"

    # Guard: Check if final_metrics.json exists
    if final_metrics_path.exists() and not force:
        msg = (
            "reports/final_metrics.json already exists. "
            "Use --force to overwrite (this is a one-time step per specification)."
        )
        logger.error(msg)
        sys.exit(2)

    if final_metrics_path.exists() and force:
        logger.warning(f"Overwriting existing {final_metrics_path}")

    # Load data and best config
    logger.info("Day 12: Final Evaluation on Test Split")
    logger.info("=" * 60)

    from src.data import load_raw, load_split, make_splits
    from src.features import build_preprocessor, split_xy
    from src.stats import bootstrap_ci

    raw_df = load_raw(cfg["paths"]["raw_csv"])
    splits = make_splits(raw_df, cfg)
    train_df = splits["train"]
    val_df = splits["val"]

    # Load test split (untouched so far)
    test_df = load_split(raw_df, cfg["paths"]["splits_dir"], "test")

    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Combine train+val into single training set
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    X_train_val, y_train_val = split_xy(train_val_df, cfg)

    logger.info(f"Combined train+val: {len(train_val_df)} rows, {X_train_val.shape[1]} features")

    # Load best config
    best_config_path = Path(cfg["paths"]["reports_dir"]) / "best_config.json"
    if not best_config_path.exists():
        msg = f"Best config not found at {best_config_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    with open(best_config_path, "r") as f:
        best_config = json.load(f)

    logger.info(f"Loaded best config: {best_config}")

    # Build and fit best model on train+val
    model = HistGradientBoostingClassifier(
        **best_config,
        random_state=cfg["seed"],
    )

    numeric_cols = cfg["numeric_cols"]
    categorical_cols = [
        c for c in X_train_val.columns
        if c not in numeric_cols and c not in cfg["drop_cols"]
    ]
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    pipeline = Pipeline([("prep", preprocessor), ("model", model)])

    logger.info("Fitting pipeline on combined train+val...")
    pipeline.fit(X_train_val, y_train_val)

    # Score on test split (ONCE)
    X_test, y_test = split_xy(test_df, cfg)

    logger.info(f"Scoring on test split: {len(test_df)} rows")
    y_test_pred_proba = pipeline.predict_proba(X_test)[:, 1]

    # Optimal threshold from Day 11 (read from config)
    optimal_threshold = cfg["metrics"]["optimal_threshold"]

    # Compute point estimates
    pr_auc_point = pr_auc(y_test, y_test_pred_proba)
    roc_auc_point = roc_auc(y_test, y_test_pred_proba)
    recall_at_precision_point = recall_at_precision(
        y_test, y_test_pred_proba, cfg["metrics"]["precision_floor"]
    )
    brier_point = brier(y_test, y_test_pred_proba)

    logger.info("Test set point estimates:")
    logger.info(f"  PR-AUC: {pr_auc_point:.4f}")
    logger.info(f"  ROC-AUC: {roc_auc_point:.4f}")
    logger.info(f"  Recall@P≥{cfg['metrics']['precision_floor']}: {recall_at_precision_point:.4f}")
    logger.info(f"  Brier: {brier_point:.4f}")

    # Bootstrap CIs (1000 resamples, percentile method)
    logger.info("Computing bootstrap CIs (1000 resamples)...")

    pr_auc_point_ci, pr_auc_ci_low, pr_auc_ci_high = bootstrap_ci(
        y_test,
        y_test_pred_proba,
        pr_auc,
        n_resamples=cfg["bootstrap"]["n_resamples"],
        alpha=cfg["bootstrap"]["alpha"],
        seed=cfg["seed"],
    )

    roc_auc_point_ci, roc_auc_ci_low, roc_auc_ci_high = bootstrap_ci(
        y_test,
        y_test_pred_proba,
        roc_auc,
        n_resamples=cfg["bootstrap"]["n_resamples"],
        alpha=cfg["bootstrap"]["alpha"],
        seed=cfg["seed"],
    )

    recall_at_precision_point_ci, recall_at_precision_ci_low, recall_at_precision_ci_high = bootstrap_ci(
        y_test,
        y_test_pred_proba,
        lambda y_t, y_s: recall_at_precision(y_t, y_s, cfg["metrics"]["precision_floor"]),
        n_resamples=cfg["bootstrap"]["n_resamples"],
        alpha=cfg["bootstrap"]["alpha"],
        seed=cfg["seed"],
    )

    brier_point_ci, brier_ci_low, brier_ci_high = bootstrap_ci(
        y_test,
        y_test_pred_proba,
        brier,
        n_resamples=cfg["bootstrap"]["n_resamples"],
        alpha=cfg["bootstrap"]["alpha"],
        seed=cfg["seed"],
    )

    # Compile results dict
    results = {
        "pr_auc": {
            "point": round(pr_auc_point_ci, 4),
            "ci_low": round(pr_auc_ci_low, 4),
            "ci_high": round(pr_auc_ci_high, 4),
        },
        "roc_auc": {
            "point": round(roc_auc_point_ci, 4),
            "ci_low": round(roc_auc_ci_low, 4),
            "ci_high": round(roc_auc_ci_high, 4),
        },
        "recall_at_precision": {
            "point": round(recall_at_precision_point_ci, 4),
            "ci_low": round(recall_at_precision_ci_low, 4),
            "ci_high": round(recall_at_precision_ci_high, 4),
        },
        "brier": {
            "point": round(brier_point_ci, 4),
            "ci_low": round(brier_ci_low, 4),
            "ci_high": round(brier_ci_high, 4),
        },
        "threshold": optimal_threshold,
        "n_test": len(y_test),
        "n_churners_test": int(np.sum(y_test)),
    }

    # Write to JSON
    final_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(final_metrics_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved final metrics to {final_metrics_path}")

    # Log results summary
    logger.info("\nFinal Evaluation Results:")
    logger.info(f"  PR-AUC: {results['pr_auc']['point']:.4f} "
                f"[{results['pr_auc']['ci_low']:.4f}, {results['pr_auc']['ci_high']:.4f}]")
    logger.info(f"  ROC-AUC: {results['roc_auc']['point']:.4f} "
                f"[{results['roc_auc']['ci_low']:.4f}, {results['roc_auc']['ci_high']:.4f}]")
    logger.info(f"  Recall@P≥{cfg['metrics']['precision_floor']}: {results['recall_at_precision']['point']:.4f} "
                f"[{results['recall_at_precision']['ci_low']:.4f}, {results['recall_at_precision']['ci_high']:.4f}]")
    logger.info(f"  Brier: {results['brier']['point']:.4f} "
                f"[{results['brier']['ci_low']:.4f}, {results['brier']['ci_high']:.4f}]")
    logger.info(f"  Test set size: {results['n_test']} (n_churners={results['n_churners_test']})")

    return results



def evaluate_split(
    pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float | None,
    cfg: dict,
) -> dict[str, Any]:
    """Evaluate a pipeline on a data split with multiple metrics.

    Computes PR-AUC, ROC-AUC, recall at precision floor, and Brier score.
    Does NOT fit the pipeline; assumes it is already fitted.

    Args:
        pipeline: Fitted sklearn Pipeline object.
        X: Feature matrix (already preprocessed or will be preprocessed by pipeline).
        y: Target labels.
        threshold: Optional decision threshold for confusion matrix.
            If None, skips confusion matrix computation.
        cfg: Config dict with keys 'metrics' containing 'precision_floor'.

    Returns:
        dict: Keys:
            - pr_auc: float
            - roc_auc: float
            - recall_at_precision: float
            - brier: float
            - confusion (optional): dict with tn, fp, fn, tp if threshold provided
    """
    # Get predictions
    y_score = pipeline.predict_proba(X)[:, 1]

    # Compute metrics
    metrics_dict = {
        "pr_auc": pr_auc(y, y_score),
        "roc_auc": roc_auc(y, y_score),
        "recall_at_precision": recall_at_precision(
            y, y_score, cfg["metrics"]["precision_floor"]
        ),
        "brier": brier(y, y_score),
    }

    # Add confusion matrix if threshold provided
    if threshold is not None:
        metrics_dict["confusion"] = confusion_at(y, y_score, threshold)

    return metrics_dict


def main() -> None:
    """CLI entry point for evaluation.

    Loads trained models, evaluates on validation set, generates metrics.csv.
    On Day 11+, also computes optimal threshold, calibration plot, and error analysis.
    On Day 12, can run final evaluation on test split with --final flag.
    """

    import joblib

    parser = argparse.ArgumentParser(description="Evaluate model on validation set")
    parser.add_argument(
        "--final",
        action="store_true",
        help="Run final evaluation on test set (Day 12 only)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run of final evaluation (overwrite existing results)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.final:
        from src.config import load_config
        cfg = load_config()
        final_evaluation(cfg, force=args.force)
        return

    # Validation evaluation (Day 8 + Day 9)
    from src.config import load_config
    from src.data import load_raw, make_splits
    from src.features import build_preprocessor, split_xy

    cfg = load_config()
    logger.info("Evaluating on validation split...")

    # Load data
    raw_df = load_raw(cfg["paths"]["raw_csv"])
    splits = make_splits(raw_df, cfg)
    train_df = splits["train"]
    val_df = splits["val"]

    X_train, y_train = split_xy(train_df, cfg)
    X_val, y_val = split_xy(val_df, cfg)
    logger.info(f"Validation set: {len(val_df)} rows, {X_val.shape[1]} features")

    # Load trained models and evaluate
    models_dir = cfg["paths"]["reports_dir"] + "/models"
    metric_results = []

    for model_name in ["dummy", "logreg", "hgb"]:
        model_path = f"{models_dir}/{model_name}.joblib"
        logger.info(f"Loading {model_name} from {model_path}...")
        pipeline = joblib.load(model_path)

        # Evaluate
        metrics = evaluate_split(pipeline, X_val, y_val, threshold=None, cfg=cfg)

        # Add model name and create result row
        result_row = {
            "model": model_name,
            "pr_auc": metrics["pr_auc"],
            "roc_auc": metrics["roc_auc"],
            "recall_at_precision": metrics["recall_at_precision"],
            "brier": metrics["brier"],
        }
        metric_results.append(result_row)

        logger.info(
            f"{model_name} validation metrics: PR-AUC={metrics['pr_auc']:.4f}, "
            f"ROC-AUC={metrics['roc_auc']:.4f}, Recall@P≥0.5={metrics['recall_at_precision']:.4f}"
        )

    # Evaluate best HGB config (Day 9)
    best_config_path = cfg["paths"]["reports_dir"] + "/best_config.json"
    if Path(best_config_path).exists():
        logger.info(f"Loading best HGB config from {best_config_path}...")
        with open(best_config_path, "r") as f:
            best_config = json.load(f)

        # Build best model
        model = HistGradientBoostingClassifier(
            **best_config,
            random_state=cfg["seed"],
        )

        # Build preprocessor
        numeric_cols = cfg["numeric_cols"]
        categorical_cols = [
            c for c in X_train.columns
            if c not in numeric_cols and c not in cfg["drop_cols"]
        ]
        preprocessor = build_preprocessor(numeric_cols, categorical_cols)

        # Build pipeline
        pipeline = Pipeline([("prep", preprocessor), ("model", model)])

        # Fit on full training data
        pipeline.fit(X_train, y_train)

        # Evaluate on validation
        metrics = evaluate_split(pipeline, X_val, y_val, threshold=None, cfg=cfg)

        # Add tuned HGB result
        result_row = {
            "model": "hgb_tuned",
            "pr_auc": metrics["pr_auc"],
            "roc_auc": metrics["roc_auc"],
            "recall_at_precision": metrics["recall_at_precision"],
            "brier": metrics["brier"],
        }
        metric_results.append(result_row)

        logger.info(
            f"hgb_tuned validation metrics: PR-AUC={metrics['pr_auc']:.4f}, "
            f"ROC-AUC={metrics['roc_auc']:.4f}, Recall@P≥0.5={metrics['recall_at_precision']:.4f}"
        )

        # Day 11: Threshold optimization, calibration, error analysis
        logger.info("Day 11: Computing optimal threshold and error analysis...")

        # Get predictions from best model
        y_val_pred_proba = pipeline.predict_proba(X_val)[:, 1]

        # Get cost parameters from config
        c_fp = cfg["cost"]["false_positive"]
        c_fn = cfg["cost"]["false_negative"]

        # Choose optimal threshold
        optimal_threshold = choose_threshold(y_val, y_val_pred_proba, c_fp, c_fn)
        logger.info(
            f"Optimal threshold: {optimal_threshold:.4f} "
            f"(c_fp={c_fp}, c_fn={c_fn})"
        )

        # Compute confusion at optimal threshold
        conf_opt = confusion_at(y_val, y_val_pred_proba, optimal_threshold)
        cost_opt = c_fp * conf_opt["fp"] + c_fn * conf_opt["fn"]
        logger.info(
            f"At optimal threshold: TP={conf_opt['tp']}, FP={conf_opt['fp']}, "
            f"FN={conf_opt['fn']}, TN={conf_opt['tn']}, cost={cost_opt:.2f}"
        )

        # Generate calibration plot
        calibration_path = cfg["paths"]["reports_dir"] + "/figures/calibration.png"
        calibration_plot(y_val, y_val_pred_proba, calibration_path)

        # Run error analysis
        from src.error_analysis import error_analysis

        error_analysis_path = cfg["paths"]["reports_dir"] + "/error_by_segment.csv"
        error_df = error_analysis(
            pipeline,
            X_val,
            y_val,
            y_val_pred_proba,
            optimal_threshold,
            cfg,
            error_analysis_path,
        )

        logger.info(f"\nError analysis summary ({len(error_df)} segments):")
        logger.info(
            error_df.groupby("segment_col")[["fp_rate", "fn_rate"]]
            .apply(lambda x: f"  mean fp_rate={x['fp_rate'].mean():.3f}, mean fn_rate={x['fn_rate'].mean():.3f}")
            .to_string()
        )

    else:
        logger.warning(f"Best config not found at {best_config_path}, skipping HGB tuned evaluation")

    # Save to metrics.csv
    metrics_df = pd.DataFrame(metric_results)
    metrics_path = cfg["paths"]["reports_dir"] + "/metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Saved validation metrics to {metrics_path}")

    print(metrics_df.to_string(index=False))



if __name__ == "__main__":
    main()
