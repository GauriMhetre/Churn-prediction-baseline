# LLD — Low-Level Design

All paths relative to repo root. Python 3.10+. Public functions need type hints and docstrings.

## 1. Config: `configs/baseline.yaml`
```yaml
seed: 42
paths:
  raw_csv: data/raw/telco_churn.csv
  db: data/churn.db
  splits_dir: data/splits
  reports_dir: reports
split:
  train: 0.60
  val: 0.20
  test: 0.20
target: Churn            # "Yes"/"No" -> 1/0
id_col: customerID
drop_cols: [customerID, gender]
numeric_cols: [tenure, MonthlyCharges, TotalCharges]
cost:
  false_positive: 1.0    # ASSUMPTION
  false_negative: 7.0    # ASSUMPTION
metrics:
  precision_floor: 0.5
bootstrap:
  n_resamples: 1000
  alpha: 0.05
cv:
  n_splits: 5
models:
  logreg: {C: 1.0, max_iter: 1000, class_weight: balanced}
  hgb:    {max_depth: 4, learning_rate: 0.1, max_iter: 200, class_weight: balanced}
```
`src/config.py`: `load_config(path: str = "configs/baseline.yaml") -> dict`
(uses `yaml.safe_load`; raises `FileNotFoundError`/`KeyError` with clear messages).

## 2. Database schema (SQLite, table `customers`)
Columns mirror the CSV, with fixes: `TotalCharges REAL NULL` (blank strings → NULL),
`SeniorCitizen INTEGER` (0/1), `churn INTEGER` (0/1) in addition to the original `Churn TEXT`.
Index on `customerID` (unique) and on `Contract`.
Use parameterized queries or fixed SQL files only; never build SQL from user input.

## 3. Modules and signatures

### `scripts/download_data.py`
`main() -> int` — download (stdlib `urllib`), print size + sha256, compare to `EXPECTED_SHA256`, exit 0/1.

### `src/data.py`
```python
def load_raw(path: str) -> pd.DataFrame
    # reads CSV; TotalCharges -> numeric (errors="coerce"); SeniorCitizen -> int;
    # adds `churn` int column; asserts unique customerID; asserts expected columns.
def to_sqlite(df: pd.DataFrame, db_path: str, table: str = "customers") -> None
    # replaces table; creates unique index on customerID.
def make_splits(df: pd.DataFrame, cfg: dict) -> dict[str, pd.DataFrame]
    # two-step stratified split on `churn` with cfg seed; returns {"train","val","test"}.
def save_split_ids(splits: dict[str, pd.DataFrame], splits_dir: str) -> None
    # writes {name}_ids.csv with a single `customerID` column.
def load_split(df: pd.DataFrame, splits_dir: str, name: str) -> pd.DataFrame
    # reads ids; raises if any ID is missing from df or if overlap exists between splits.
```
Split algorithm: first split off test (`test_size=cfg.split.test`), then split the remainder
into train/val with `val_size = cfg.split.val / (1 - cfg.split.test)`; both `stratify=churn`, `random_state=seed`.

### `src/sqlrun.py`
```python
def run_sql_file(db_path: str, sql_path: str) -> pd.DataFrame
def run_all(db_path: str, sql_dir: str, out_dir: str) -> dict[str, pd.DataFrame]
    # executes each sql/*.sql (single SELECT per file or `-- name:` separated blocks),
    # writes reports/sql_<name>.csv.
```
Required queries (minimum 6):
1. `profile`: row count, distinct customers, null counts per column, overall churn rate.
2. `duplicates`: duplicate `customerID` check.
3. `churn_by_contract`: churn rate and n by `Contract`.
4. `churn_by_tenure_bucket`: buckets 0–12, 13–24, 25–48, 49+ months.
5. `churn_by_internet_payment`: churn rate by `InternetService` × `PaymentMethod` (n ≥ 30 only).
6. `rank_segments`: window function `RANK() OVER (ORDER BY churn_rate DESC)` over contract × tenure bucket.

### `src/stats.py`
```python
def wilson_ci(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float]
    # statsmodels.stats.proportion.proportion_confint(method="wilson")
def segment_churn_table(df: pd.DataFrame, col: str, alpha: float = 0.05) -> pd.DataFrame
    # columns: segment, n, churners, rate, ci_low, ci_high
def chi2_test(df: pd.DataFrame, col: str, target: str = "churn") -> dict
    # {"chi2","p_value","dof","cramers_v"} via scipy.stats.chi2_contingency
def bootstrap_ci(y_true, y_score, metric_fn, n_resamples: int, alpha: float, seed: int) -> tuple[float, float, float]
    # returns (point_estimate, low, high); percentile method; resample rows with replacement.
```
Report effect size (Cramér's V) next to every p-value.

### `src/features.py`
```python
def build_preprocessor(cfg: dict, df_columns: list[str]) -> ColumnTransformer
    # numeric: SimpleImputer(median) -> StandardScaler
    # categorical (all remaining object cols after drop): SimpleImputer(most_frequent)
    #   -> OneHotEncoder(handle_unknown="ignore", drop="if_binary")
    # columns in cfg["drop_cols"] and target are excluded.
def build_pipeline(cfg: dict, model, df_columns: list[str]) -> Pipeline
    # Pipeline([("prep", preprocessor), ("model", model)])
def split_xy(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.Series]
```
Rule: `fit` is only ever called on a Pipeline with training rows.

### `src/train.py`
```python
def get_models(cfg: dict) -> dict[str, estimator]
    # {"dummy": DummyClassifier(strategy="prior"),
    #  "logreg": LogisticRegression(**cfg.models.logreg, random_state=seed),
    #  "hgb": HistGradientBoostingClassifier(**cfg.models.hgb, random_state=seed)}
def cross_validate_model(name: str, pipeline: Pipeline, X_train, y_train, cfg: dict) -> pd.DataFrame
    # StratifiedKFold(shuffle=True, random_state=seed); per-fold PR-AUC, ROC-AUC; returns fold table.
def fit_and_save(name: str, pipeline: Pipeline, X, y, out_dir: str) -> str
    # joblib.dump to reports/models/<name>.joblib (gitignored); returns path.
def log_run(run: dict, path: str = "reports/runs.csv") -> None
    # appends one row: timestamp, model, params, cv metrics.
def main(config_path: str = "configs/baseline.yaml") -> None   # `make train`
```
`DummyClassifier(strategy="prior")` yields constant probabilities; PR-AUC then equals the positive rate — this is the baseline to beat.

### `src/evaluate.py`
```python
def pr_auc(y_true, y_score) -> float            # average_precision_score
def roc_auc(y_true, y_score) -> float
def recall_at_precision(y_true, y_score, floor: float) -> float
    # max recall over thresholds where precision >= floor; 0.0 if none.
def brier(y_true, y_score) -> float
def confusion_at(y_true, y_score, threshold: float) -> dict   # tn, fp, fn, tp
def choose_threshold(y_true, y_score, c_fp: float, c_fn: float) -> float
    # scan unique score thresholds; minimize c_fp*FP + c_fn*FN; ties -> higher threshold.
def calibration_plot(y_true, y_score, path: str) -> None      # matplotlib, 10 quantile bins
def evaluate_split(pipeline, X, y, threshold: float | None, cfg: dict) -> dict
def final_evaluation(cfg: dict, force: bool = False) -> dict
    # 1. refuse if reports/final_metrics.json exists and not force (exit code 2, clear message)
    # 2. refit best pipeline on train+val; threshold fixed from validation run
    # 3. score test once; bootstrap CIs for PR-AUC and recall@precision
    # 4. write reports/final_metrics.json
def main() -> None    # CLI: `python -m src.evaluate [--final] [--force]`
```
`make eval` without `--final` evaluates on validation only.

## 4. Makefile targets
```
data   -> python scripts/download_data.py
db     -> python -m src.data --build-db
sql    -> python -m src.sqlrun
stats  -> python -m src.stats
split  -> python -m src.data --split
train  -> python -m src.train
eval   -> python -m src.evaluate            # validation
final  -> python -m src.evaluate --final    # Day 12 only
test   -> pytest -q
lint   -> ruff check .
clean  -> rm -f data/*.db
```
Recipe lines use real TAB characters.

## 5. Outputs
- `reports/sql_*.csv`, `reports/runs.csv`, `reports/metrics.csv`, `reports/final_metrics.json`
- `reports/figures/*.png` (calibration, churn-by-segment with CIs, PR curve)
- `reports/models/*.joblib` (gitignored)

## 6. CI: `.github/workflows/ci.yml`
Trigger: push, pull_request. Steps: checkout → setup-python (3.11) with pip cache →
`pip install -r requirements.txt` → `ruff check .` → `pytest -q`.
Top-level `permissions: contents: read`. No secrets used.

## 7. Error handling rules
- Fail fast with explicit messages on: missing columns, duplicate IDs, split overlap, checksum mismatch, second final eval.
- No bare `except`. No silent fallbacks that change results.
- Use `logging`, not `print`, inside `src/` (scripts may print).
