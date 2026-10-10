"""Feature engineering and preprocessing pipeline."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> ColumnTransformer:
    """Build a ColumnTransformer for numeric and categorical preprocessing.

    Numeric path: SimpleImputer(strategy="median") + StandardScaler
    Categorical path: SimpleImputer(strategy="most_frequent") + OneHotEncoder(
        handle_unknown="ignore", drop="if_binary")

    Args:
        numeric_cols: List of numeric column names.
        categorical_cols: List of categorical column names (will include
            SeniorCitizen as string).

    Returns:
        ColumnTransformer with 'numeric' and 'categorical' transformers.
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="if_binary",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_cols),
            ("categorical", categorical_transformer, categorical_cols),
        ]
    )

    return preprocessor


def split_xy(
    df, cfg: dict
) -> tuple:
    """Split dataframe into features and target.

    Args:
        df: DataFrame with all columns.
        cfg: Config dict with keys 'target', 'id_col', 'drop_cols'.

    Returns:
        (X, y) where X has drop_cols and target removed, y is the binary target.
    """
    target_col = cfg["target"]
    drop_cols = cfg["drop_cols"]

    # Remove drop_cols and both target columns (original + binary) from X
    # The binary target column is always named 'churn' (lowercase) if it exists
    cols_to_drop = [c for c in (drop_cols + [target_col, "churn"]) if c in df.columns]
    X = df.drop(columns=cols_to_drop)

    # Extract y as the binary target column (prefer 'churn' if it exists, else use target_col)
    if "churn" in df.columns:
        y = df["churn"]
    else:
        y = df[target_col]

    return X, y
