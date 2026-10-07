# Customer Churn Prediction Baseline

**Status:** Day 1 of 14

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

## 4. Setup & Quickstart

### Prerequisites
- Python 3.10+
- `make` (or execute scripts directly)

### Reproduce Setup
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
4. Download data and verify SHA-256 checksum:
   ```bash
   make data
   ```

### Manual Download Fallback
If the automated download fails:
1. Download `Telco-Customer-Churn.csv` from:
   `https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv`
2. Save to `data/raw/telco_churn.csv`.
3. Verify SHA-256 checksum:
   `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`

## 5. Results Table
*Model training and evaluation will be populated on Days 8–12.*

| Model | PR-AUC (Val) | ROC-AUC (Val) | Recall @ Prec ≥ 0.5 | Cost / Customer |
|---|---|---|---|---|
| *Dummy Baseline (TBD Day 8)* | - | - | - | - |
| *Logistic Regression (TBD Day 8)* | - | - | - | - |
| *HistGradientBoosting (TBD Day 9)* | - | - | - | - |

## 6. License
MIT License. See `LICENSE` for details.
