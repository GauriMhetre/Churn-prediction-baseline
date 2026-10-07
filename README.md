# Customer Churn Prediction Baseline

**Status:** Day 2 of 14

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

## 5. Setup & Quickstart

### Prerequisites
- Python 3.10+
- `make` (or execute python modules directly)

### Reproduce Setup & Database Build
1. Clone the repository:
   ```bash
   git clone https://github.com/GauriMhetre/Churn-prediction-baseline.git
   cd Churn-prediction-baseline
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .\.venv\Scripts\activate
   ```
3. Install pinned dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Download data, load into SQLite, and run SQL profiling:
   ```bash
   make data
   make db
   make sql
   ```

### Manual Download Fallback
If the automated download fails:
1. Download `Telco-Customer-Churn.csv` from:
   `https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv`
2. Save to `data/raw/telco_churn.csv`.
3. Verify SHA-256 checksum:
   `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`

## 6. Results Table
*Model training and evaluation will be populated on Days 8–12.*

| Model | PR-AUC (Val) | ROC-AUC (Val) | Recall @ Prec ≥ 0.5 | Cost / Customer |
|---|---|---|---|---|
| *Dummy Baseline (TBD Day 8)* | - | - | - | - |
| *Logistic Regression (TBD Day 8)* | - | - | - | - |
| *HistGradientBoosting (TBD Day 9)* | - | - | - | - |

## 7. License
MIT License. See `LICENSE` for details.
