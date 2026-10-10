"""Model training and cross-validation module."""

import argparse
import json
import logging
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.config import load_config
from src.data import load_raw, make_splits
from src.evaluate import pr_auc, roc_auc
from src.features import build_preprocessor, split_xy

logger = logging.getLogger(__name__)


def get_models(cfg: dict) -> dict[str, Any]:
    """Build dictionary of model estimators from config.

    Creates three baseline models:
    - DummyClassifier (prior strategy): predicts class distribution
    - LogisticRegression: linear model with balanced class weights
    - HistGradientBoostingClassifier: gradient boosting with balanced weights

    Args:
        cfg: Config dict with 'seed' and 'models' sections.

    Returns:
        dict: Keys {dummy, logreg, hgb} with fitted estimator objects.
    """
    seed = cfg["seed"]

    models = {
        "dummy": DummyClassifier(strategy="prior", random_state=seed),
        "logreg": LogisticRegression(
            **cfg["models"]["logreg"],
            random_state=seed,
        ),
        "hgb": HistGradientBoostingClassifier(
            **cfg["models"]["hgb"],
            random_state=seed,
        ),
    }

    return models


def cross_validate_model(
    name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cfg: dict,
) -> pd.DataFrame:
    """Run stratified cross-validation on a pipeline.

    Splits training data into k folds using StratifiedKFold, trains the pipeline
    on each fold's training portion, and evaluates on each fold's validation portion.
    Computes PR-AUC and ROC-AUC for each fold.

    Args:
        name: Model name (for logging).
        pipeline: sklearn Pipeline with preprocessor and model.
        X_train: Feature matrix for training.
        y_train: Target labels for training.
        cfg: Config dict with 'cv' section containing 'n_splits' and 'seed'.

    Returns:
        pd.DataFrame: One row per fold with columns:
            - fold: fold number (0-indexed)
            - pr_auc: PR-AUC on validation fold
            - roc_auc: ROC-AUC on validation fold
    """
    seed = cfg["seed"]
    n_splits = cfg["cv"]["n_splits"]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    results = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_val = y_train.iloc[val_idx]

        # Fit pipeline on training fold
        pipeline.fit(X_fold_train, y_fold_train)

        # Evaluate on validation fold
        y_score = pipeline.predict_proba(X_fold_val)[:, 1]
        fold_pr_auc = pr_auc(y_fold_val, y_score)
        fold_roc_auc = roc_auc(y_fold_val, y_score)

        results.append(
            {
                "fold": fold_idx,
                "pr_auc": fold_pr_auc,
                "roc_auc": fold_roc_auc,
            }
        )

        logger.info(
            f"{name} fold {fold_idx}: PR-AUC={fold_pr_auc:.4f}, "
            f"ROC-AUC={fold_roc_auc:.4f}"
        )

    return pd.DataFrame(results)


def grid_search_hgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cfg: dict,
) -> pd.DataFrame:
    """Grid search for HistGradientBoostingClassifier hyperparameters.

    Generates a grid of ≤12 configurations combining max_depth, learning_rate,
    and max_iter. For each configuration, performs 5-fold stratified cross-validation
    and computes mean/std PR-AUC and ROC-AUC. Results are logged to reports/runs.csv.

    Grid design:
    - max_depth: [3, 4, 5, 6]
    - learning_rate: [0.05, 0.1, 0.15, 0.2]
    - max_iter: [100, 150, 200, 300]
    Selected to keep total ≤12 configs (3 x 2 x 2 = 12).

    Args:
        X_train: Feature matrix for training.
        y_train: Target labels for training.
        cfg: Config dict with 'seed', 'cv', and 'paths' sections.

    Returns:
        pd.DataFrame: Grid search results with columns:
            - config_id: config identifier (0-indexed)
            - max_depth: max_depth parameter
            - learning_rate: learning_rate parameter
            - max_iter: max_iter parameter
            - mean_pr_auc: mean CV PR-AUC
            - std_pr_auc: std CV PR-AUC
            - mean_roc_auc: mean CV ROC-AUC
            - std_roc_auc: std CV ROC-AUC
    """
    seed = cfg["seed"]
    preprocessor = build_preprocessor(cfg["numeric_cols"], [
        c for c in X_train.columns
        if c not in cfg["numeric_cols"] and c not in cfg["drop_cols"]
    ])

    # Define grid (limited to ≤12 configs)
    max_depths = [3, 4, 5, 6]
    learning_rates = [0.05, 0.1, 0.15]
    max_iters = [100, 200, 300]

    # Generate all combinations
    grid_configs = list(product(max_depths, learning_rates, max_iters))
    logger.info(f"Grid search: {len(grid_configs)} configs to evaluate")

    results = []

    for config_id, (max_depth, learning_rate, max_iter) in enumerate(grid_configs):
        logger.info(
            f"\nConfig {config_id + 1}/{len(grid_configs)}: "
            f"max_depth={max_depth}, learning_rate={learning_rate}, max_iter={max_iter}"
        )

        # Build model with current config
        model = HistGradientBoostingClassifier(
            max_depth=max_depth,
            learning_rate=learning_rate,
            max_iter=max_iter,
            class_weight="balanced",
            random_state=seed,
        )

        # Build pipeline
        pipeline = Pipeline([("prep", preprocessor), ("model", model)])

        # Cross-validate
        cv_results = cross_validate_model(
            f"hgb_config_{config_id}",
            pipeline,
            X_train,
            y_train,
            cfg,
        )

        # Compute mean and std metrics
        mean_pr_auc = cv_results["pr_auc"].mean()
        mean_roc_auc = cv_results["roc_auc"].mean()
        std_pr_auc = cv_results["pr_auc"].std()
        std_roc_auc = cv_results["roc_auc"].std()

        logger.info(
            f"Config {config_id}: PR-AUC={mean_pr_auc:.4f}±{std_pr_auc:.4f}, "
            f"ROC-AUC={mean_roc_auc:.4f}±{std_roc_auc:.4f}"
        )

        # Log to runs.csv
        run_record = {
            "model": f"hgb_config_{config_id}",
            "params": {
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                "max_iter": max_iter,
                "class_weight": "balanced",
            },
            "mean_pr_auc": mean_pr_auc,
            "mean_roc_auc": mean_roc_auc,
            "std_pr_auc": std_pr_auc,
            "std_roc_auc": std_roc_auc,
        }
        log_run(run_record, cfg["paths"]["reports_dir"] + "/runs.csv")

        # Store result
        results.append({
            "config_id": config_id,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "max_iter": max_iter,
            "mean_pr_auc": mean_pr_auc,
            "std_pr_auc": std_pr_auc,
            "mean_roc_auc": mean_roc_auc,
            "std_roc_auc": std_roc_auc,
        })

    return pd.DataFrame(results)


def select_best_config(grid_results_df: pd.DataFrame) -> dict[str, Any]:
    """Select best HGB config based on highest mean CV PR-AUC.

    Finds the configuration with the highest mean PR-AUC and saves it to
    reports/best_config.json as a JSON dict.

    Args:
        grid_results_df: Grid search results DataFrame with at least
            max_depth, learning_rate, max_iter, mean_pr_auc columns.

    Returns:
        dict: Best configuration with keys:
            - max_depth
            - learning_rate
            - max_iter
            - class_weight ("balanced")
    """
    # Find config with highest mean PR-AUC (first in order if tie)
    best_idx = grid_results_df["mean_pr_auc"].idxmax()
    best_row = grid_results_df.loc[best_idx]

    best_config = {
        "max_depth": int(best_row["max_depth"]),
        "learning_rate": float(best_row["learning_rate"]),
        "max_iter": int(best_row["max_iter"]),
        "class_weight": "balanced",
    }

    logger.info(f"Best config selected: {best_config}")
    logger.info(f"  Mean PR-AUC: {best_row['mean_pr_auc']:.4f}")
    logger.info(f"  Mean ROC-AUC: {best_row['mean_roc_auc']:.4f}")

    return best_config


def fit_and_save(
    name: str,
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    out_dir: str,
) -> str:
    """Fit a pipeline and save to joblib file.

    Args:
        name: Model name.
        pipeline: sklearn Pipeline to fit and save.
        X: Feature matrix for fitting.
        y: Target labels for fitting.
        out_dir: Directory to save models (e.g., "reports/models").

    Returns:
        str: Path to saved model file.
    """
    out_path = Path(out_dir) / f"{name}.joblib"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Fit pipeline on full data
    pipeline.fit(X, y)

    # Save
    joblib.dump(pipeline, out_path)
    logger.info(f"Saved {name} model to {out_path}")

    return str(out_path)


def log_run(run: dict, path: str = "reports/runs.csv") -> None:
    """Append a run record to the runs log CSV file.

    Appends one row with run metadata: timestamp, model, parameters, and metrics.
    Creates file if it does not exist.

    Args:
        run: Dictionary with run metadata:
            - model: model name
            - params: model parameters (dict)
            - mean_pr_auc: mean CV PR-AUC
            - mean_roc_auc: mean CV ROC-AUC
            - std_pr_auc: std CV PR-AUC
            - std_roc_auc: std CV ROC-AUC
        path: CSV file path (default: reports/runs.csv).
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare row
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": run["model"],
        "params": str(run["params"]),
        "mean_pr_auc": run["mean_pr_auc"],
        "mean_roc_auc": run["mean_roc_auc"],
        "std_pr_auc": run["std_pr_auc"],
        "std_roc_auc": run["std_roc_auc"],
    }

    df_row = pd.DataFrame([row])

    if out_path.exists():
        df_row.to_csv(out_path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(out_path, mode="w", header=True, index=False)

    logger.info(f"Logged run to {path}")


def main() -> None:
    """CLI entry point for training.

    Loads config, loads raw data, creates splits, builds preprocessing pipeline,
    trains models with cross-validation, and logs results.
    """
    parser = argparse.ArgumentParser(description="Train and cross-validate models")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    cfg = load_config(args.config)
    logger.info(f"Loaded config from {args.config}")

    # Truncate/reset runs.csv at start to avoid stale entries
    runs_csv_path = cfg["paths"]["reports_dir"] + "/runs.csv"
    Path(runs_csv_path).parent.mkdir(parents=True, exist_ok=True)
    # Start fresh: write only the header
    runs_df = pd.DataFrame()
    runs_df.to_csv(runs_csv_path, index=False)
    logger.info(f"Reset runs.csv at {runs_csv_path}")

    # Load data
    raw_df = load_raw(cfg["paths"]["raw_csv"])
    logger.info(f"Loaded {len(raw_df)} raw records")

    # Create splits
    splits = make_splits(raw_df, cfg)
    train_df = splits["train"]
    val_df = splits["val"]

    logger.info(
        f"Train/val sizes: {len(train_df)}/{len(val_df)}"
    )

    # Split features and target
    X_train, y_train = split_xy(train_df, cfg)
    _, y_val = split_xy(val_df, cfg)

    logger.info(
        f"Features: {X_train.shape[1]}, Target churn rate: "
        f"{y_train.mean():.1%} (train), {y_val.mean():.1%} (val)"
    )

    # Identify numeric and categorical columns
    numeric_cols = cfg["numeric_cols"]
    all_feature_cols = [c for c in X_train.columns if c not in cfg["drop_cols"]]
    categorical_cols = [c for c in all_feature_cols if c not in numeric_cols]

    # Build preprocessor
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    # Get models
    models = get_models(cfg)

    # Train and evaluate each model
    for model_name, model in models.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {model_name}...")
        logger.info(f"{'='*60}")

        # Build pipeline
        pipeline = Pipeline([("prep", preprocessor), ("model", model)])

        # Cross-validate
        cv_results = cross_validate_model(
            model_name,
            pipeline,
            X_train,
            y_train,
            cfg,
        )

        # Log results
        mean_pr_auc = cv_results["pr_auc"].mean()
        mean_roc_auc = cv_results["roc_auc"].mean()
        std_pr_auc = cv_results["pr_auc"].std()
        std_roc_auc = cv_results["roc_auc"].std()

        logger.info(
            f"\n{model_name} CV results:"
        )
        logger.info(f"  PR-AUC:  {mean_pr_auc:.4f} ± {std_pr_auc:.4f}")
        logger.info(f"  ROC-AUC: {mean_roc_auc:.4f} ± {std_roc_auc:.4f}")

        # Get model params
        params = model.get_params()

        # Log run
        run_record = {
            "model": model_name,
            "params": params,
            "mean_pr_auc": mean_pr_auc,
            "mean_roc_auc": mean_roc_auc,
            "std_pr_auc": std_pr_auc,
            "std_roc_auc": std_roc_auc,
        }
        log_run(run_record, cfg["paths"]["reports_dir"] + "/runs.csv")

        # Fit and save final model on full training data
        fit_and_save(
            model_name,
            pipeline,
            X_train,
            y_train,
            cfg["paths"]["reports_dir"] + "/models",
        )

    logger.info(f"\n{'='*60}")
    logger.info("Training complete. Results logged to reports/runs.csv")
    logger.info("Models saved to reports/models/")
    logger.info(f"{'='*60}")

    # Grid search for HGB hyperparameters
    logger.info(f"\n{'='*60}")
    logger.info("Starting HGB hyperparameter grid search...")
    logger.info(f"{'='*60}")

    grid_results = grid_search_hgb(X_train, y_train, cfg)

    # Select best config
    best_config = select_best_config(grid_results)

    # Save best config to JSON
    best_config_path = Path(cfg["paths"]["reports_dir"]) / "best_config.json"
    best_config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(best_config_path, "w") as f:
        json.dump(best_config, f, indent=2)
    logger.info(f"Saved best HGB config to {best_config_path}")

    logger.info(f"\n{'='*60}")
    logger.info("Grid search complete. Best config:")
    logger.info(json.dumps(best_config, indent=2))
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
