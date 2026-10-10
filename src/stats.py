"""Statistical analysis module for churn predictions."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.proportion import proportion_confint

from src.config import load_config
from src.data import load_raw, load_split

logger = logging.getLogger(__name__)


def wilson_ci(
    successes: int, n: int, alpha: float = 0.05
) -> tuple[float, float]:
    """Calculate Wilson confidence interval for a proportion.

    Args:
        successes: Number of successes (e.g., churners).
        n: Total number of observations.
        alpha: Significance level (default 0.05 for 95% CI).

    Returns:
        tuple: (lower_bound, upper_bound) of the confidence interval.
    """
    if n == 0:
        return (0.0, 1.0)
    if successes < 0 or successes > n:
        raise ValueError(
            f"successes={successes} must be in [0, {n}]"
        )

    low, high = proportion_confint(
        successes, n, alpha=alpha, method="wilson"
    )
    return (low, high)


def segment_churn_table(
    df: pd.DataFrame, col: str, alpha: float = 0.05
) -> pd.DataFrame:
    """Compute churn rate and Wilson CI for each segment of a column.

    Args:
        df: DataFrame with 'churn' column and segment column.
        col: Name of the segment column (e.g., 'Contract').
        alpha: Significance level for CI.

    Returns:
        pd.DataFrame with columns:
        segment, n, churners, rate, ci_low, ci_high
    """
    results = []

    for segment_val in sorted(df[col].unique()):
        segment_df = df[df[col] == segment_val]
        n = len(segment_df)
        churners = segment_df["churn"].sum()
        rate = churners / n if n > 0 else 0.0

        ci_low, ci_high = wilson_ci(int(churners), n, alpha=alpha)

        results.append(
            {
                "segment": segment_val,
                "n": n,
                "churners": int(churners),
                "rate": round(rate, 4),
                "ci_low": round(ci_low, 4),
                "ci_high": round(ci_high, 4),
            }
        )

    return pd.DataFrame(results)


def plot_segment_churn(
    df: pd.DataFrame, col: str, title: str, out_path: str
) -> None:
    """Plot churn rate by segment with Wilson CI error bars.

    Args:
        df: DataFrame with segment churn analysis (from segment_churn_table).
        col: Column name for x-axis labels.
        title: Plot title.
        out_path: Path to save the PNG file.
    """
    _fig, ax = plt.subplots(figsize=(10, 6))

    segments = df["segment"].astype(str)
    rates = df["rate"]
    ci_low = df["ci_low"]
    ci_high = df["ci_high"]

    # Error bars: distance from point to CI bounds
    errors = [rates - ci_low, ci_high - rates]

    ax.bar(segments, rates, yerr=errors, capsize=5, alpha=0.7, color="steelblue")
    ax.set_ylabel("Churn Rate", fontsize=12)
    ax.set_xlabel(col, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylim([0, 1])
    ax.grid(axis="y", alpha=0.3)

    # Rotate x labels if many segments
    if len(segments) > 5:
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=100)
    plt.close()
    logger.info(f"Saved plot to {out_path}")


def chi2_test(
    df: pd.DataFrame, col: str, target: str = "churn"
) -> dict:
    """Perform chi-square test of independence between a categorical column and target.

    Args:
        df: DataFrame with target column.
        col: Name of the categorical column to test.
        target: Name of the target column (default "churn").

    Returns:
        dict with keys: chi2, p_value, dof, cramers_v
    """
    # Create contingency table
    contingency = pd.crosstab(df[col], df[target])

    # Perform chi-square test
    chi2, p_value, dof, _expected_freq = chi2_contingency(contingency)

    # Compute Cramér's V (effect size)
    n = contingency.sum().sum()
    min_dim = min(contingency.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0.0

    return {
        "chi2": round(chi2, 4),
        "p_value": round(p_value, 6),
        "dof": int(dof),
        "cramers_v": round(cramers_v, 4),
    }


def compute_vif(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    """Compute variance inflation factors for numeric columns.

    Uses statsmodels add_constant to add an intercept column, then computes VIF
    for each numeric column (excluding the constant). This matches standard
    practice in OLS regression.

    Args:
        df: DataFrame with numeric columns.
        numeric_cols: List of numeric column names.

    Returns:
        pd.DataFrame with columns: variable, vif
    """
    from statsmodels.tools.tools import add_constant

    # Handle NaN values: drop rows with any NaN in numeric cols
    df_clean = df[numeric_cols].dropna()

    if len(df_clean) == 0 or len(numeric_cols) == 0:
        return pd.DataFrame({"variable": numeric_cols, "vif": [np.nan] * len(numeric_cols)})

    # Add constant (intercept) column
    df_with_const = add_constant(df_clean)

    # Compute VIF for each column (excluding the constant)
    vif_data = []
    for i, col in enumerate(numeric_cols):
        try:
            vif = variance_inflation_factor(
                df_with_const.values, i + 1  # +1 because constant is at index 0
            )
            vif_data.append({"variable": col, "vif": round(vif, 2)})
        except (ValueError, np.linalg.LinAlgError):
            # Handle singular matrix or other numerical issues
            vif_data.append({"variable": col, "vif": np.nan})

    return pd.DataFrame(vif_data)


def compute_correlation(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    """Compute pairwise correlations among numeric columns.

    Args:
        df: DataFrame with numeric columns.
        numeric_cols: List of numeric column names.

    Returns:
        pd.DataFrame with columns: var1, var2, correlation
    """
    corr_matrix = df[numeric_cols].corr()

    # Extract upper triangle (avoid duplicates)
    corr_pairs = []
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            corr_pairs.append(
                {
                    "var1": numeric_cols[i],
                    "var2": numeric_cols[j],
                    "correlation": round(corr_matrix.iloc[i, j], 4),
                }
            )

    return pd.DataFrame(corr_pairs)


def bootstrap_ci(
    y_true: np.ndarray | list,
    y_score: np.ndarray | list,
    metric_fn,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval for a metric using percentile method.

    Resamples (y_true, y_score) pairs with replacement n_resamples times,
    computes metric for each resample, and returns (point_estimate, ci_low, ci_high).

    Args:
        y_true: True binary labels (0 or 1).
        y_score: Predicted probabilities (0 to 1).
        metric_fn: Callable that takes (y_true, y_score) and returns float.
        n_resamples: Number of bootstrap resamples (default 1000).
        alpha: Significance level for two-tailed CI (default 0.05).
        seed: Random seed for reproducibility.

    Returns:
        tuple: (point_estimate, ci_low, ci_high)
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    n = len(y_true)
    rng = np.random.RandomState(seed)

    # Compute point estimate
    point_estimate = metric_fn(y_true, y_score)

    # Generate bootstrap samples
    bootstrap_metrics = []
    for _ in range(n_resamples):
        indices = rng.choice(n, size=n, replace=True)
        y_true_boot = y_true[indices]
        y_score_boot = y_score[indices]
        try:
            metric_val = metric_fn(y_true_boot, y_score_boot)
            bootstrap_metrics.append(metric_val)
        except ValueError as e:
            # Skip resamples that fail (e.g., all same class)
            logger.debug(f"Skipping bootstrap resample due to error: {e}")

    bootstrap_metrics = np.array(bootstrap_metrics)

    # Compute percentile CI
    ci_low = np.percentile(bootstrap_metrics, 100 * alpha / 2)
    ci_high = np.percentile(bootstrap_metrics, 100 * (1 - alpha / 2))

    return (float(point_estimate), float(ci_low), float(ci_high))


def main() -> None:
    """CLI entrypoint for statistical analysis."""
    logging.basicConfig(level=logging.INFO)
    cfg = load_config()

    # Load raw data and train split
    logger.info("Loading raw data and train split...")
    raw_df = load_raw(cfg["paths"]["raw_csv"])
    train_df = load_split(
        raw_df, cfg["paths"]["splits_dir"], "train"
    )

    logger.info(f"Loaded train split with {len(train_df)} rows")

    # ============ Day 5: Segment churn (Wilson CIs) ============
    logger.info("=" * 60)
    logger.info("DAY 5: Segment Churn Analysis (TRAIN split)")
    logger.info("=" * 60)

    # Compute segment churn for each column
    segment_cols = ["Contract", "InternetService"]
    all_segments = []

    for col in segment_cols:
        logger.info(f"Computing segment churn for {col}...")
        segment_df = segment_churn_table(train_df, col)
        segment_df["column"] = col
        all_segments.append(segment_df)

        # Plot
        title = f"Churn Rate by {col} (Train Split with 95% Wilson CI)"
        out_path = (
            f"{cfg['paths']['reports_dir']}/figures/churn_by_{col}.png"
        )
        plot_segment_churn(segment_df, col, title, out_path)

    # Add tenure bucket segments (create temp column for bucketing)
    logger.info("Computing segment churn for tenure buckets...")
    train_df_copy = train_df.copy()
    train_df_copy["tenure_bucket"] = pd.cut(
        train_df_copy["tenure"],
        bins=[0, 12, 24, 48, 100],
        labels=["0-12mo", "13-24mo", "25-48mo", "49+mo"],
        right=False,
    )
    tenure_segment_df = segment_churn_table(train_df_copy, "tenure_bucket")
    tenure_segment_df["column"] = "tenure_bucket"
    all_segments.append(tenure_segment_df)

    # Plot tenure buckets
    title = "Churn Rate by Tenure Bucket (Train Split with 95% Wilson CI)"
    out_path = f"{cfg['paths']['reports_dir']}/figures/churn_by_tenure_bucket.png"
    plot_segment_churn(tenure_segment_df, "tenure_bucket", title, out_path)

    # Combine all segments and save
    combined_df = pd.concat(all_segments, ignore_index=True)
    out_csv = f"{cfg['paths']['reports_dir']}/stats_segments.csv"
    combined_df.to_csv(out_csv, index=False)
    logger.info(f"Saved segment stats to {out_csv}")

    # ============ Day 6: Statistical tests (Chi-square, VIF) ============
    logger.info("=" * 60)
    logger.info("DAY 6: Statistical Tests (TRAIN split)")
    logger.info("=" * 60)

    # Chi-square tests on categorical columns
    cat_cols = ["Contract", "InternetService", "PaymentMethod", "PhoneService"]
    chi2_results = []

    for col in cat_cols:
        if col in train_df.columns:
            logger.info(f"Running chi-square test for {col}...")
            result = chi2_test(train_df, col, target="churn")
            result["variable"] = col
            chi2_results.append(result)

    chi2_df = pd.DataFrame(chi2_results)

    # Compute VIF for numeric columns
    numeric_cols = cfg["numeric_cols"]
    logger.info(f"Computing VIF for numeric columns: {numeric_cols}...")
    vif_df = compute_vif(train_df, numeric_cols)
    logger.info(f"VIF results:\n{vif_df.to_string(index=False)}")

    # Compute correlations among numeric columns
    logger.info("Computing correlations among numeric columns...")
    corr_df = compute_correlation(train_df, numeric_cols)
    logger.info(f"Correlation results:\n{corr_df.to_string(index=False)}")

    # Save chi-square results
    out_csv = f"{cfg['paths']['reports_dir']}/stats_tests.csv"
    chi2_df.to_csv(out_csv, index=False)
    logger.info(f"Saved chi-square tests to {out_csv}")

    # Save VIF and correlations as separate reports for reference
    vif_path = f"{cfg['paths']['reports_dir']}/stats_vif.csv"
    vif_df.to_csv(vif_path, index=False)
    logger.info(f"Saved VIF analysis to {vif_path}")

    corr_path = f"{cfg['paths']['reports_dir']}/stats_correlations.csv"
    corr_df.to_csv(corr_path, index=False)
    logger.info(f"Saved correlation analysis to {corr_path}")

    print("Statistical analysis complete.")
    print(f"  - Segment stats: {cfg['paths']['reports_dir']}/stats_segments.csv")
    print(f"  - Chi-square tests: {out_csv}")
    print(f"  - VIF collinearity: {vif_path}")
    print(f"  - Correlations: {corr_path}")


if __name__ == "__main__":
    main()
