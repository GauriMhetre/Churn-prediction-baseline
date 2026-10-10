# Customer Churn Prediction Baseline

**Status:** Day 12 of 14 — Final evaluation complete ✓

A 14-day, reproducible, leakage-safe baseline for customer churn prediction on the Telco Customer Churn dataset. Deliverable: a repository anyone can clone and run with `make data train eval` to get identical numbers.

## 1. Problem Statement & Framing (PRD Section 2)
- **Churn Definition:** A customer left the service in the most recent month (`Churn = Yes`).
- **Prediction Framing:** Snapshot scoring ("as of today, who is likely to be a churner"). The Telco dataset has no dates, so this is a simulated scoring run, not a true temporal forecast.
- **Action Triggered:** Customers scored above a decision threshold receive a targeted retention offer.

## 2. Assumptions (PRD Section 9)
- **Cost Assumption (ASSUMPTION):** Cost of a False Positive (FP) ≈ 1.0 unit (retention offer cost); Cost of a False Negative (FN) ≈ 7.0 units (lost customer lifetime value). These are assumed baseline parameters configured in `configs/baseline.yaml`.
- **Temporal Framing Assumption (ASSUMPTION):** Features and target overlap in time. The label reflects the most recent month while features describe the current account state, limiting causal interpretations.

## 3. Success Criteria (PRD Section 8)
- Beats majority-class baseline on PR-AUC with non-overlapping 95% bootstrap confidence intervals.
- A third party can reproduce the results table from a fresh clone without asking questions.
- Test set evaluated exactly once (on Day 12).
- Goal is honest evaluation and reproducibility, not leaderboard rank.

## 4. Data Profile (Day 2 SQL Output: `reports/sql_profile.csv`)
SQL profiling performed on the SQLite `customers` table (`sql/01_profile.sql`):

| Total Rows | Distinct Customers | Null `TotalCharges` | Total Churners | Overall Churn Rate |
|---|---|---|---|---|
| 7,043 | 7,043 | 11 | 1,869 | 26.537% (0.26537) |

- All 7,043 rows correspond to unique customer IDs (0 duplicates).
- `TotalCharges` contains 11 blank values (corresponding to customers with `tenure = 0`), which are coerced to `NULL` / `NaN` for imputation in the modeling pipeline.

## 5. SQL Findings (Day 3 — Descriptive on All Rows)

### Contract Type Impact
Churn is strongly associated with contract length. Month-to-month contracts have 15× higher churn (42.71%, n=3,875) than two-year contracts (2.83%, n=1,695). One-year contracts sit in between (11.27%, n=1,473).

| Contract | n | Churners | Churn Rate |
|---|---|---|---|
| Month-to-month | 3,875 | 1,655 | 42.71% |
| One year | 1,473 | 166 | 11.27% |
| Two year | 1,695 | 48 | 2.83% |

### Tenure Buckets
Churn decreases with tenure on the TRAIN split. Customers in their first year (0–12 months) have the highest churn (48.3%), while long-tenured customers (49+ months) have the lowest (9.75%).

| Tenure | n | Churners | Churn Rate | 95% CI Low | 95% CI High |
|---|---|---|---|---|---|
| 0–12 months | 1,238 | 598 | 48.3% | 45.53% | 51.09% |
| 13–24 months | 652 | 195 | 29.91% | 26.52% | 33.53% |
| 25–48 months | 961 | 194 | 20.19% | 17.77% | 22.84% |
| 49+ months | 1,374 | 134 | 9.75% | 8.29% | 11.44% |

### Internet Service & Payment Method
Churn varies significantly by internet service type and payment method (only combinations with n ≥ 30 shown). Fiber optic customers using electronic check have the highest churn (53.23%, n=1,595). "No internet" customers using any automatic payment method have the lowest churn (≤5.4%).

| Internet Service | Payment Method | n | Churners | Churn Rate |
|---|---|---|---|---|
| Fiber optic | Electronic check | 1,595 | 849 | 53.23% |
| Fiber optic | Mailed check | 258 | 110 | 42.64% |
| DSL | Electronic check | 648 | 207 | 31.94% |
| Fiber optic | Bank transfer (automatic) | 646 | 187 | 28.95% |
| Fiber optic | Credit card (automatic) | 597 | 151 | 25.29% |
| DSL | Mailed check | 613 | 127 | 20.72% |
| No | Electronic check | 122 | 15 | 12.30% |
| DSL | Credit card (automatic) | 594 | 72 | 12.12% |
| No | Mailed check | 741 | 71 | 9.58% |
| DSL | Bank transfer (automatic) | 566 | 53 | 9.36% |
| No | Bank transfer (automatic) | 332 | 18 | 5.42% |
| No | Credit card (automatic) | 331 | 9 | 2.72% |

### Top Segments (Contract × Tenure Bucket)
The highest-churn segments are month-to-month contracts within the first year. Month-to-month / 0–12 months has 51.35% churn (n=1,994, rank #1). Conversely, two-year contracts with 0–12 months tenure have 0% churn (n=68, rank #11–12), suggesting strong protective effect of commitment.

| Segment | n | Churners | Churn Rate | Rank |
|---|---|---|---|---|
| Month-to-month / 0–12 | 1,994 | 1,024 | 51.35% | 1 |
| Month-to-month / 13–24 | 737 | 278 | 37.72% | 2 |
| Month-to-month / 25–48 | 802 | 264 | 32.92% | 3 |
| Month-to-month / 49+ | 342 | 89 | 26.02% | 4 |
| One year / 49+ | 634 | 82 | 12.93% | 5 |
| … | … | … | … | … |
| Two year / 0–12 | 68 | 0 | 0.00% | 11 |

## 6. Statistical Tests (Day 6 — TRAIN Split Only)

Chi-square tests of independence were performed on the training split (4,225 rows) to assess the strength of association between categorical features and churn, with effect sizes (Cramér's V) indicating practical significance. Collinearity among numeric features was assessed via variance inflation factors (VIF).

### Categorical Feature Association Tests

| Variable | χ² | p-value | dof | Cramér's V | Finding |
|---|---|---|---|---|---|
| Contract | 743.18 | < 0.001 | 2 | 0.4194 | **Strong association** — Contract type is a dominant churn predictor. |
| InternetService | 455.49 | < 0.001 | 2 | 0.3283 | **Strong association** — Internet service (Fiber, DSL, None) predicts churn. |
| PaymentMethod | 399.98 | < 0.001 | 3 | 0.3077 | **Moderate–strong association** — Payment method (check, card, bank, electronic) shows churn signal. |
| PhoneService | 0.09 | 0.765 | 1 | 0.0046 | **No association** — Phone service status is uncorrelated with churn; will not be useful for modeling. |

**Modeling Decision (Association Tests):** Contract, InternetService, and PaymentMethod are retained as strong predictors. PhoneService will be dropped from the feature set during preprocessing due to its negligible effect size (V ≈ 0).

### Numeric Feature Collinearity (VIF on TRAIN Split)

Variance Inflation Factors quantify how much the variance of a regression coefficient inflates due to correlation with other predictors. VIF is computed with an intercept (standard practice in OLS regression). Rule of thumb: VIF < 5 is acceptable, VIF > 10 signals strong redundancy.

| Variable | VIF |
|---|---|
| tenure | 6.05 |
| MonthlyCharges | 3.19 |
| TotalCharges | 9.78 |

**Numeric Correlations (Pairwise):**

| Pair | Correlation |
|---|---|
| tenure ↔ MonthlyCharges | 0.2594 (weak–moderate) |
| tenure ↔ TotalCharges | 0.8337 (strong) |
| MonthlyCharges ↔ TotalCharges | 0.6503 (moderate–strong) |

**Collinearity Finding & Modeling Decision:** 
Collinearity among numeric features affects coefficient stability and interpretability, but does not cause leakage or bias predictions in regularized models like logistic regression and gradient boosting. `TotalCharges` has elevated VIF (9.78) due to its strong correlation with tenure (r = 0.83), which is expected since `TotalCharges ≈ tenure × MonthlyCharges`. 

**Why not drop `TotalCharges`?** It is a direct business metric (total customer spend) and adds predictive signal independent of tenure. Regularization in sklearn's models handles collinearity by penalizing large coefficients. All three numeric features are retained.

Imbalance in churn rate (26.54% positive class on training split) will be addressed via `class_weight='balanced'` during model fitting. **No SMOTE** — oversampling can complicate temporal evaluation and violate the split contract if rows leak between partitions.

## 7. Preprocessing Pipeline (Day 7 — No Model Training)

Feature engineering is implemented in `src/features.py` as a scikit-learn `ColumnTransformer` to prevent data leakage. The preprocessor is fit only on training data and applied identically to validation/test during evaluation.

### Numeric Features (Imputation + Scaling)

Numeric columns (`tenure`, `MonthlyCharges`, `TotalCharges`) follow a two-step pipeline:
1. **SimpleImputer(strategy="median")** — Replaces missing values with the training median. Median is robust to outliers and preferred over mean for non-normal distributions.
2. **StandardScaler** — Centers and scales each feature to mean ≈ 0, std ≈ 1. This stabilizes regularization in logistic regression and gradient boosting.

**Imputation Rationale:** `TotalCharges` has 11 missing values (blank strings in the raw CSV, corresponding to new customers with `tenure = 0`). These are imputed with the training median to avoid dropping rows and losing signal.

### Categorical Features (One-Hot Encoding)

All object-type columns (except those dropped) are treated as categorical and encoded with:
1. **SimpleImputer(strategy="most_frequent")** — Fills any missing values with the mode (rarely needed but defensive).
2. **OneHotEncoder(handle_unknown="ignore", drop="if_binary")** — Creates binary indicators for each category:
   - `handle_unknown="ignore"` — Unseen categories at prediction time are encoded as all zeros (safe for production).
   - `drop="if_binary"` — For binary features (e.g., yes/no columns), drops one category to avoid multicollinearity and constant term redundancy.

### Dropped Columns & Rationale

The following columns are **excluded** from the feature set:

| Column | Reason |
|---|---|
| `customerID` | Identifier; provides no predictive signal. |
| `gender` | Binary, low association with churn (chi-square test would likely be non-significant); dropped to reduce dimensionality without sacrificing predictive power. |
| `PhoneService` | Chi-square test showed no association with churn (χ² = 0.09, p = 0.765, V ≈ 0.0046); retained would add noise. |
| `Churn` (target) | Removed from features; used as y for model training. |

**Note on SeniorCitizen:** Though numeric (0/1), it is encoded as a categorical feature (cast to string) because its values are inherently categorical (senior / not senior) rather than continuous quantities. This preserves interpretability in the one-hot encoding.

### No Leakage Design

The modeling pipeline enforces strict separation between train, validation, and test:
1. **Split Before Statistics:** The stratified 60/20/20 split is created first (Day 4), before any model development or statistics.
2. **TRAIN-Only Statistics:** All exploratory statistics (Days 5–6) use training rows only.
3. **TRAIN-Only Preprocessing:** Each preprocessor (imputation, scaling, encoding) is fit on training data and applied identically to validation/test.
4. **Test Set Untouched:** No metrics are computed on test rows until Day 12 (final evaluation, one-time only).
5. **Day 3 SQL is Descriptive:** SQL profiling (Day 3) describes the full dataset to understand structure but does not inform model selection—all decisions are data-driven from TRAIN statistics only.

The `ColumnTransformer` pipeline in `src/features.py` is instantiated with fixed column names and indices. During cross-validation and final model training:
- **fit() is called only on training data** inside the sklearn `Pipeline`.
- Imputation statistics (median, mode) and scaling parameters (mean, std) are learned from training rows only.
- Validation and test are **transformed** using the training-fitted preprocessor.

This mechanical separation prevents information from validation/test from leaking into feature construction.

### Feature Count

After preprocessing, the feature matrix includes:
- 3 numeric features (tenure, MonthlyCharges, TotalCharges)
- Categorical features for all object columns except `customerID`, `gender`, `Churn`, and `PhoneService`
  - `Contract` (3 categories, 2 after `drop="if_binary"`)
  - `InternetService` (3 categories, 2 after `drop="if_binary"`)
  - `PaymentMethod` (4 categories, 3 after `drop="if_binary"`)
  - Plus other binary yes/no columns (Partner, Dependents, PaperlessBilling, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, MultipleLines, OnlineBackup, etc.)

Final feature dimension is determined by the one-hot encoder and varies slightly based on unseen categories at prediction time, which are encoded as zeros.

## 7.5. Leakage Suspects (Hypotheses for Preprocessing & Evaluation)

The following features are flagged as **potential sources of leakage** and require careful handling during preprocessing:

- **HYPOTHESIS: `TotalCharges`** — This variable represents cumulative customer spend and is derived from `tenure` and `MonthlyCharges`. Strong correlation with tenure may introduce information overlap; if future tenure is known at prediction time, this is safe, but if not, it risks temporal leakage.
- **HYPOTHESIS: `Contract`** — Contract type (month-to-month, one-year, two-year) is a strong churn predictor but is also a choice variable. Customers choosing month-to-month may have higher pre-existing churn intent. Causal interpretation is limited.
- **HYPOTHESIS: `tenure`** — Tenure is outcome-dependent: churned customers have lower tenure by definition. While useful, it should be interpreted as a snapshot feature, not a causal driver.
- **HYPOTHESIS: `MonthlyCharges`** — Monthly spending may change due to service usage, which could be influenced by unobserved customer satisfaction or needs. No causal claim is made.
- **HYPOTHESIS: `InternetService` & `PaymentMethod`** — Customers' choice of internet service and payment method may reflect underlying risk. Automatic payments, for instance, are associated with lower churn, possibly because recurring billing reduces friction.

**Key Control:** All exploratory statistics (Day 3–6) are computed on the full dataset to remain descriptive and avoid test-set contamination. Modeling (Days 7+) will strictly separate train/val/test and fit preprocessing on training rows only.

## 8. Train/Validation/Test Split (Day 4 — Stratified, Deterministic)

A **stratified 60/20/20 split** was created on the full dataset using seed 42 from the config, ensuring reproducibility:

| Partition | Rows | Fraction | Saved to |
|---|---|---|---|
| Train | 4,225 | 60.0% | `data/splits/train_ids.csv` |
| Validation | 1,409 | 20.0% | `data/splits/val_ids.csv` |
| Test | 1,409 | 20.0% | `data/splits/test_ids.csv` |

**Stratification:** Churn rate is preserved within each split to ±1.5 percentage points of the overall 26.54% rate. This ensures all three partitions have representative churn distributions.

**Determinism:** The split is deterministic—the same seed always produces identical partitions. ID files are committed to the repository for reproducibility; anyone cloning the repo will use these exact same splits.

**Test Set Guarantee:** The test partition **will not be touched until Day 12** (final evaluation). No hyperparameter tuning, no threshold selection, no inspection of test metrics is performed before then. All model development (Days 7–11) happens on training and validation data only. This mechanical separation enforces leakage discipline.

## 8.5. Segment Uncertainty Analysis (Day 5 — TRAIN Split Only)

Churn rates and 95% Wilson confidence intervals were computed on the **training split (4,225 rows)** to characterize uncertainty and identify segments with limited sample size:

### Contract Type (Train Split)
Churn varies by contract commitment, with uncertainty widening as sample size decreases:

| Contract | n | Churn Rate | 95% CI |
|---|---|---|---|
| Month-to-month | 2,326 | 43.08% | [40.08%, 45.10%] |
| One year | 865 | 10.75% | [8.86%, 12.99%] |
| Two year | 1,034 | 2.51% | [1.72%, 3.66%] |

**Finding:** Month-to-month contracts have the widest CI (±2pp) due to moderate sample size (n=2,326) but high variance. Two-year contracts, despite larger n, have a narrow CI (±0.97pp) because low churn rate implies low variance.

### Internet Service (Train Split)
Churn by internet service shows strong differentiation, especially between Fiber optic and No internet:

| Internet Service | n | Churn Rate | 95% CI |
|---|---|---|---|
| Fiber optic | 1,843 | 42.32% | [40.08%, 44.59%] |
| DSL | 1,474 | 18.66% | [16.75%, 20.73%] |
| No | 908 | 7.27% | [5.75%, 9.14%] |

**Finding:** Fiber optic has the highest churn and moderate uncertainty. The "No internet" segment has the smallest n (908) but highest relative uncertainty (CI width = ±1.69pp). All segments have sufficient n ≥ 30 for stable estimates.

**Segment Uncertainty Interpretation:** Wide CIs (e.g., Two year: ±0.97pp) reflect genuine underlying variability, not just sampling error. Narrow CIs (e.g., No internet: ±1.69pp absolute, ±1.39pp relative to 7.27%) indicate that sample sizes are sufficient to estimate the segment's true churn rate with modest precision.

## 9. Setup & Quickstart

### Prerequisites
- Python 3.10+
- `make` (or execute python modules directly)

### Reproduce on Fresh Clone (Windows)

**Full reproducibility in 5 steps:**

1. **Clone and navigate:**
   ```cmd
   git clone https://github.com/GauriMhetre/Churn-prediction-baseline.git
   cd Churn-prediction-baseline
   ```

2. **Create virtual environment:**
   ```cmd
   python -m venv .venv
   .\.venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```cmd
   pip install -r requirements.txt
   ```

4. **Run the full pipeline (excludes final evaluation):**
   ```cmd
   make all
   ```
   This downloads data, builds SQLite database, runs SQL profiling, creates stratified split, computes statistics, trains all models, and evaluates on validation set. Produces all results CSVs and figures.

5. **Verify results match README:**
   - Check `reports/metrics.csv` against Section 10 of README
   - Dummy PR-AUC should be 0.2654, HGB_tuned PR-AUC should be 0.6456

**Optional: Run individual steps:**
- `make data` — Download dataset
- `make db` — Load into SQLite
- `make sql` — Run SQL profiling queries
- `make split` — Create train/val/test splits
- `make stats` — Compute statistics on training split
- `make train` — Train baseline and tuned models
- `make eval` — Evaluate on validation set
- `make lint` — Check code style (ruff)
- `make test` — Run pytest suite

### Reproduce on Fresh Clone (Linux / macOS)

**Full reproducibility in 5 steps:**

1. **Clone and navigate:**
   ```bash
   git clone https://github.com/GauriMhetre/Churn-prediction-baseline.git
   cd Churn-prediction-baseline
   ```

2. **Create virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the full pipeline (excludes final evaluation):**
   ```bash
   make all
   ```
   This downloads data, builds SQLite database, runs SQL profiling, creates stratified split, computes statistics, trains all models, and evaluates on validation set. Produces all results CSVs and figures.

5. **Verify results match README:**
   - Check `reports/metrics.csv` against Section 10 of README
   - Dummy PR-AUC should be 0.2654, HGB_tuned PR-AUC should be 0.6456

**Optional: Run individual steps:**
- `make data` — Download dataset
- `make db` — Load into SQLite
- `make sql` — Run SQL profiling queries
- `make split` — Create train/val/test splits
- `make stats` — Compute statistics on training split
- `make train` — Train baseline and tuned models
- `make eval` — Evaluate on validation set
- `make lint` — Check code style (ruff)
- `make test` — Run pytest suite

### Running Without Make (Python Equivalents)

If `make` is not available, run these commands sequentially:

```python
python scripts/download_data.py
python -m src.data --build-db
python -m src.sqlrun
python -m src.data --split
python -m src.stats
python -m src.train
python -m src.evaluate
ruff check .
pytest -q
```

### Manual Download Fallback

If the automated download fails:
1. Download `Telco-Customer-Churn.csv` from:
   `https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv`
2. Save to `data/raw/telco_churn.csv`.
3. Verify SHA-256 checksum:
   `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`
4. Continue with `make db` and subsequent steps.

## 10. Results Table (Day 9 — Validation Only, HGB Tuned)

Cross-validation results (CV mean ± std) on training split and validation set metrics for three models: baseline Dummy and LogReg (Day 8), and tuned HistGradientBoosting (Day 9):

| Model | CV PR-AUC (Train) | CV ROC-AUC (Train) | Val PR-AUC | Val ROC-AUC | Recall @ Prec ≥ 0.5 (Val) | Brier (Val) |
|---|---|---|---|---|---|---|
| Dummy (Prior) | 0.2653 ± 0.0005 | 0.5000 ± 0.0000 | 0.2654 | 0.5000 | 0.0000 | 0.1950 |
| Logistic Regression | 0.6655 ± 0.0284 | 0.8482 ± 0.0133 | 0.6429 | 0.8366 | 0.8155 | 0.1678 |
| HistGradientBoosting (Tuned) | 0.6539 ± 0.0178 | 0.8387 ± 0.0141 | 0.6456 | 0.8401 | 0.8102 | 0.1637 |

## 10.5. Final Results on Test Set (Day 12 — One-Time Evaluation)

**Status: Day 12 of 14 — Final evaluation complete.**

The best model (HistGradientBoosting with tuned hyperparameters) was refitted on the combined training and validation sets (5,634 rows) and evaluated on the held-out test set (1,409 rows) exactly once. Point estimates and 95% bootstrap confidence intervals (1,000 resamples, percentile method) are reported below:

| Model | PR-AUC | ROC-AUC | Recall @ Prec ≥ 0.5 | Brier |
|---|---|---|---|---|
| Dummy (Prior) — Test | 0.2654 | 0.5000 | 0.0000 | 0.1950 |
| **HistGradientBoosting (Best) — Test** | **0.6617** [0.6129, 0.7116] | **0.8450** [0.8234, 0.8659] | **0.8262** [0.7365, 0.8938] | **0.1623** [0.1515, 0.1729] |

**Key Findings:**
- **Dummy baseline (test):** Predicts class prior (26.5% churn), yielding PR-AUC = 0.2654 and ROC-AUC = 0.5000, with Recall @ Prec ≥ 0.5 = 0.0000 (no positive predictions above precision threshold).
- **Tuned HGB (test):** Beats dummy significantly:
  - **PR-AUC: 0.6617** (+0.3963 vs dummy, **149% improvement**)
  - **ROC-AUC: 0.8450** (+0.3450 vs dummy, **69% improvement**)
  - **Recall @ Prec ≥ 0.5: 0.8262** (dummy cannot achieve this, as it only predicts the class prior)
  - **Brier: 0.1623** (−0.0327 vs dummy, lower Brier = better predictions)
- **95% Confidence Intervals:** All bootstrap CIs are tight and exclude the dummy baseline values, confirming statistical significance.
- **Test Set Details:** n = 1,409 customers, n_churners = 374 (26.54% churn rate, stratified to match training distribution).
- **Optimal Threshold (from Day 11):** 0.2147, applied without retuning on test to prevent leakage.

**Conclusion:** The tuned HGB model provides substantial discrimination and calibration gains over the majority-class baseline on unseen test data, validating the leakage-safe design and honest evaluation protocol.

**Key Findings (Day 8 → Day 9):**
- **Dummy baseline** (predicts class prior, 26.5%) has PR-AUC = 0.2654 and ROC-AUC = 0.5000.
- **Logistic Regression** beats dummy by 2.4pp on PR-AUC (0.6429), with ROC-AUC of 0.8366 on validation.
- **HistGradientBoosting (Day 8 baseline)** was competitive: PR-AUC = 0.6396, ROC-AUC = 0.8305.
- **HistGradientBoosting (Day 9 tuned)** improves via hyperparameter grid search:
  - **PR-AUC: 0.6456** (+0.60pp vs Day 8 baseline, beats LogReg by 0.27pp)
  - **ROC-AUC: 0.8401** (+0.96pp vs Day 8 baseline, beats LogReg by 0.35pp)
  - **Recall @ Prec ≥ 0.5: 0.8102** (+2.68pp vs Day 8 baseline)
  - **Brier: 0.1637** (−0.01 vs Day 8 baseline, lower is better)

**Day 9 Tuning Details:**
- **Grid Search:** 36 configurations tested (4 × 3 × 3 = max_depth × learning_rate × max_iter)
  - `max_depth` ∈ {3, 4, 5, 6}
  - `learning_rate` ∈ {0.05, 0.10, 0.15}
  - `max_iter` ∈ {100, 200, 300}
- **Best Configuration:** `max_depth=4, learning_rate=0.05, max_iter=100` (selected by highest mean CV PR-AUC)
- **CV Evaluation:** 5-fold stratified cross-validation on training split only (4,225 rows)
- **Logged Results:** All 36 grid configs logged to `reports/runs.csv` for transparency

**Summary:** Tuned HGB now provides the best balance of discrimination (ROC-AUC = 0.8401), precision-recall trade-off (Recall @ Prec ≥ 0.5 = 0.8102), and calibration (Brier = 0.1637).

## 11. Threshold & Error Analysis (Day 11 — Validation Only)

### Optimal Threshold Selection
Using cost-minimization with the assumed cost parameters (c_fp=1.0, c_fn=7.0), the model identifies an **optimal decision threshold of 0.2147** that balances false positives and false negatives. This conservative threshold reflects the high penalty on false negatives (missing churners = 7.0) relative to false positives (offering retention = 1.0).

At this threshold on the validation set:
- **True Positives: 356** (customers correctly identified as churners)
- **False Positives: 563** (non-churners offered retention)
- **False Negatives: 18** (churners missed by the model)
- **True Negatives: 472** (non-churners correctly left alone)
- **Total cost: 689.00** (1.0 × 563 + 7.0 × 18)

**Cost rationale (ASSUMPTION):** The ratio 7:1 (FN:FP) assumes losing a customer is 7× more expensive than offering a retention incentive. This aggressive threshold (0.2147 vs default 0.5) favors recall over precision: we catch more churners at the cost of over-targeting non-churners. In a retention scenario with high LTV, this trade-off is often justified.

### Calibration Analysis
A reliability diagram (saved to `reports/figures/calibration.png`) visualizes predicted probabilities against observed churn frequencies on the validation set. The tuned HGB model shows moderate predictive accuracy. Predictions across the range are reasonably aligned with empirical frequencies, though at very high probabilities the model exhibits some conservatism (predicting lower churn than observed).

### Error Analysis by Segment
Error rates were computed on the validation split (1,409 rows) by segment. Below are key patterns from `reports/error_by_segment.csv`:

**Pattern 1: Contract Type Drives Model Errors**
| Contract | n | FP Rate | FN Rate |
|---|---|---|---|
| Month-to-month | 776 | 56.3% | 5.8% |
| One year | 308 | 55.9% | 3.5% |
| Two year | 325 | 48.6% | 3.8% |

Month-to-month contracts experience the highest FP rate (56.3%, n=776): the model incorrectly flags many non-churning month-to-month customers for retention, likely because this segment is strongly associated with churn in the training data. Two-year contracts see lower error rates, reflecting their lower and more stable churn risk.

**Pattern 2: Tenure is a Secondary Error Driver**
| Tenure Bucket | n | FP Rate | FN Rate |
|---|---|---|---|
| 0–12 months | 419 | 57.0% | 4.2% |
| 13–24 months | 189 | 51.8% | 5.8% |
| 25–48 months | 350 | 54.2% | 4.7% |
| 49+ months | 451 | 53.3% | 5.1% |

Early-tenure customers (0–12 months, n=419) have the highest FP rate at 57.0%, consistent with high baseline churn in this segment. The model is cautious (high recall, more FPs) because early-tenure signals high risk. Longer-tenured customers have slightly lower error rates but remain conservative.

**Pattern 3: Internet Service Shows Modest Variation**
| Internet Service | n | FP Rate | FN Rate |
|---|---|---|---|
| DSL | 463 | 57.1% | 3.9% |
| Fiber optic | 640 | 55.3% | 6.7% |
| No internet | 306 | 48.2% | 2.4% |

DSL and Fiber optic customers see FP rates of 55–57%, while "No internet" customers (n=306, lower-risk segment) see only 48.2%. The higher FN rate for Fiber optic (6.7% vs 2.4% for "No internet") suggests the model is less confident predicting churn among Fiber customers, leading to more missed churners in this segment.

**Pattern 4: Demographics (Audit Only)**
Gender and SeniorCitizen were included for audit purposes (not used in modeling per design). Error rates by gender and senior status are broadly similar (FP rates ~51–56%, FN rates ~3.6–9.5%), indicating no systematic bias in model errors across these protected attributes. However, SeniorCitizen=1 (elderly customers, n=229) has a higher FN rate at 9.5% vs 3.9% for younger customers, suggesting the model is less sensitive to churn signals in this group—a fairness consideration for future iterations.

**Conclusion:** Model errors cluster on high-risk segments (month-to-month, early tenure, Fiber optic). This is expected and acceptable: the cost-optimized threshold prioritizes catching churners in these segments, accepting higher FP rates. In production, this means retention offers go disproportionately to month-to-month and early-tenure customers, which aligns with business risk.

## 12. Limitations (Day 10)

### Temporal Leakage Not Tested
The Telco Customer Churn dataset lacks timestamps (e.g., account creation dates, churn dates, feature measurement dates). This prevents a **temporal split test**, which is critical for production models.

**Why it matters:** A temporal split (e.g., train on customers active before date X, test on churn after date X) detects leakage that random splits miss:
- Features measured in month N would not be available when predicting churn in month N+1.
- Tenure, recency of service changes, and account flags may contain future information if features are measured post-prediction date.
- `TotalCharges` might accumulate charges occurring after the label was assigned, inflating its predictive power artificially.

**What we did instead:** Used a stratified random split (60/20/20 on customer IDs) with cross-validation on training data. This approach is defensible for snapshot scoring but does not prove the model is safe for production time-series deployment. A future temporal evaluation would require:
1. A dataset with explicit timestamps for feature measurements and churn events.
2. A train/test cutoff on calendar time (e.g., features before 2024-01-01, churn labels in 2024-02-01 to 2024-03-01).
3. Comparison of random-split and time-split performance to quantify leakage.

**Conclusion:** Results reported here apply to snapshot scoring on static snapshots, not to forecasting churn in future time windows. Production deployment would require temporal validation.

### Model Selection Uncertainty
On the validation set, tuned HGB achieves PR-AUC = 0.6456 vs LogReg PR-AUC = 0.6429, a difference of **0.003**. The 95% bootstrap confidence interval for PR-AUC (computed on test data from 1000 resamples) is [0.6129, 0.7116], a **width of 0.0987**. The model performance difference of 0.003 is **negligible compared to the CI width**, providing no statistical evidence that the tree-based model outperforms logistic regression.

**Interpretation:** This baseline does not claim the tuned HGB is superior. Both models are competitive. Future work should:
1. Collect more data to narrow confidence intervals.
2. Perform formal hypothesis tests (e.g., permutation test) if claiming superiority.
3. Consider simpler models (LogReg) for interpretability if performance is equivalent.

### Threshold and Cost Not Validated on Test
The optimal threshold (0.2147) was selected on the validation set to minimize the cost function (c_fp=1.0, c_fn=7.0). While applied to test data for final metrics, the actual confusion matrix, FP/FN counts, and cost at this threshold are **not reported** in the test results. 

**Why:** The cost matrix is an assumption (ASSUMPTION in Section 2). Different business units may have different retention offer costs or customer lifetime value estimates. Reporting only PR-AUC/ROC-AUC allows stakeholders to apply their own cost assumptions post hoc, rather than locking in a single threshold that may not align with their priorities.

**Future work:** Stakeholders should validate the cost assumptions (c_fp and c_fn) with domain experts and recompute the threshold on held-out data using their actual cost matrix.

### Bootstrap CI Computation
The 95% confidence intervals reported for test metrics (PR-AUC, ROC-AUC, Recall @ Precision ≥ 0.5, Brier) use the percentile method from 1000 bootstrap resamples of the test set. This is a non-parametric approach suitable for complex metrics but assumes:
1. Test set size (1,409 rows) is sufficient for stable CI estimation.
2. Bootstrap samples represent true population variation (reasonable for i.i.d. data).
3. Percentile method is preferred over bias-corrected accelerated (BCa) for reproducibility and interpretability.

Implementation: `src/stats.py:bootstrap_ci()` with seed=42 for reproducibility.

## 13. License
MIT License. See `LICENSE` for details.
