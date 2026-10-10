"""Error analysis and segmentation module."""

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)


def create_tenure_buckets(df: pd.DataFrame, tenure_col: str = "tenure") -> pd.Series:
    """Create tenure buckets from tenure column.

    Buckets: [0-12], [13-24], [25-48], [49+]

    Args:
        df: DataFrame with tenure column.
        tenure_col: Name of tenure column.

    Returns:
        pd.Series: Tenure bucket labels.
    """
    bins = [0, 13, 25, 49, 1000]
    labels = ["0-12", "13-24", "25-48", "49+"]
    return pd.cut(df[tenure_col], bins=bins, labels=labels, right=False)


def error_analysis(
    pipeline: Any,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    y_val_pred_proba: np.ndarray,
    threshold: float,
    cfg: dict,
    output_path: str,
) -> pd.DataFrame:
    """Compute error rates (FP/FN) by segment on validation split.

    For each segment column (Contract, tenure_bucket, InternetService,
    SeniorCitizen, gender), computes FP_rate and FN_rate for each category.

    Args:
        pipeline: Fitted sklearn Pipeline (not used directly; X_val is preprocessed).
        X_val: Validation feature matrix with segment columns.
        y_val: Validation labels.
        y_val_pred_proba: Predicted probabilities (shape n,).
        threshold: Decision threshold.
        cfg: Config dict with 'drop_cols' for reference.
        output_path: Path to save results CSV.

    Returns:
        pd.DataFrame: Segment analysis with columns:
            segment_col, segment_value, n, churners, fp, fn, fp_rate, fn_rate
    """
    # Load raw data to get segment columns (since X_val may have dropped some)
    from src.data import load_raw, load_split

    raw_df = load_raw(cfg["paths"]["raw_csv"])
    val_df = load_split(
        raw_df,
        cfg["paths"]["splits_dir"],
        "val",
    )

    # Reset indices to align with y_val (which is reset_index(drop=True) from split_xy)
    val_df_reset = val_df.reset_index(drop=True)
    y_val_reset = y_val.reset_index(drop=True) if isinstance(y_val, pd.Series) else y_val

    # Get predictions at threshold
    y_pred = (y_val_pred_proba >= threshold).astype(int)

    # Compute confusion matrix to validate
    _, fp_total, fn_total, _ = confusion_matrix(
        y_val_reset, y_pred, labels=[0, 1]
    ).ravel()
    logger.info(
        f"Validation confusion at threshold {threshold}: "
        f"FP={fp_total}, FN={fn_total}"
    )

    results = []

    # Segment columns to analyze
    segment_cols = {
        "Contract": val_df_reset["Contract"],
        "InternetService": val_df_reset["InternetService"],
        "SeniorCitizen": val_df_reset["SeniorCitizen"].astype(str),
        "gender": val_df_reset["gender"],
        "tenure_bucket": create_tenure_buckets(val_df_reset),
    }

    for seg_col_name, seg_col in segment_cols.items():
        # Reset index on segment column to align with y_val_reset
        seg_col_reset = seg_col.reset_index(drop=True)

        # Get unique values in this segment
        unique_values = seg_col_reset.unique()

        for segment_value in unique_values:
            # Filter to this segment
            mask = seg_col_reset == segment_value
            n_segment = np.sum(mask)

            if n_segment == 0:
                continue

            y_seg = y_val_reset[mask] if isinstance(y_val_reset, pd.Series) else y_val_reset[mask.values]
            y_pred_seg = y_pred[mask] if isinstance(y_pred, pd.Series) else y_pred[mask.values]

            # Ensure we have numpy arrays for confusion_matrix
            if isinstance(y_seg, pd.Series):
                y_seg = y_seg.values
            if isinstance(y_pred_seg, pd.Series):
                y_pred_seg = y_pred_seg.values

            # Compute confusion for segment
            tn_seg, fp_seg, fn_seg, tp_seg = confusion_matrix(
                y_seg, y_pred_seg, labels=[0, 1]
            ).ravel()

            # Compute rates
            # FP_rate = FP / (FP + TN) = FP / negatives
            fp_rate = fp_seg / (fp_seg + tn_seg) if (fp_seg + tn_seg) > 0 else 0.0

            # FN_rate = FN / (FN + TP) = FN / positives
            fn_rate = fn_seg / (fn_seg + tp_seg) if (fn_seg + tp_seg) > 0 else 0.0

            churners = np.sum(y_seg == 1)

            results.append(
                {
                    "segment_col": seg_col_name,
                    "segment_value": str(segment_value),
                    "n": int(n_segment),
                    "churners": int(churners),
                    "fp": int(fp_seg),
                    "fn": int(fn_seg),
                    "fp_rate": float(fp_rate),
                    "fn_rate": float(fn_rate),
                }
            )

            logger.debug(
                f"{seg_col_name}={segment_value}: n={n_segment}, "
                f"churners={churners}, FP_rate={fp_rate:.3f}, FN_rate={fn_rate:.3f}"
            )

    # Create DataFrame and save
    error_df = pd.DataFrame(results)

    # Sort for readability
    error_df = error_df.sort_values(
        by=["segment_col", "segment_value"]
    ).reset_index(drop=True)

    # Ensure output directory exists
    from pathlib import Path

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    error_df.to_csv(output_path, index=False)
    logger.info(f"Saved error analysis to {output_path}")

    return error_df


