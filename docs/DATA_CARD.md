# DATA_CARD.md — Telco Customer Churn

## Source
Public sample dataset (IBM sample data), downloaded by `scripts/download_data.py`
to `data/raw/telco_churn.csv`. License/terms: check the source repository before redistributing; this repo does not commit the data.

## Shape (verify with your own run; do not trust these numbers blindly)
Expected ≈ 7,043 rows × 21 columns; churn rate ≈ 26–27%.

## Columns (expected)
| Group | Columns |
|---|---|
| Identifier | `customerID` |
| Demographics | `gender`, `SeniorCitizen`, `Partner`, `Dependents` |
| Account | `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges` |
| Services | `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` |
| Target | `Churn` (Yes/No) |

## Known quirks
- `TotalCharges` has blank strings for a handful of customers with `tenure = 0`; read as numeric with coercion, then impute inside the pipeline.
- `TotalCharges ≈ tenure × MonthlyCharges` (collinear); note when interpreting logistic coefficients.
- Many service columns contain "No internet service"/"No phone service" in addition to Yes/No; treat as categories, not noise.
- `SeniorCitizen` is 0/1 integer while other binaries are Yes/No strings.

## Leakage and framing risks
- No timestamps: cannot do a temporal split or verify the label was unknown at prediction time.
- Label means "left in the most recent month" while features describe the current account state; some fields (e.g. tenure, charges) may partially reflect the outcome. List suspects in the README after Day 3.

## Fairness notes
- `gender` is excluded from modeling by design (D8). `SeniorCitizen` retained only if justified; audit error rates by both in Day 11.

## Intended use / not intended for
Learning and portfolio demonstration only. Not for real customer decisions.

## Optional second dataset (Day 10)
**Not used.** Temporal leakage was not tested because the Telco dataset lacks timestamps. A time-aware dataset would require explicit dates for feature measurements and churn events to enable a temporal split (train before cutoff date, test after cutoff) and quantify leakage risk. Production deployment of this model should include temporal validation on a time-stamped dataset before scoring real customers.

See README §11 (Limitations) for details on what a temporal split would test and why it matters.
