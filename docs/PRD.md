# PRD — Customer Churn Prediction: Reproducible Baseline

| Field | Value |
|---|---|
| Owner | Gauri M |
| Repo | https://github.com/GauriMhetre/Churn-prediction-baseline |
| Version | 1.0 (Week 1–2 of a 12-week ML Engineer sprint) |
| Status | In progress (Day 1 of 14) |

## 1. Overview
A telecom provider loses customers each month. This project builds a reproducible,
leakage-safe baseline that scores customers by churn risk so a retention team can
prioritize offers. The repo itself is the product: it must be re-runnable by a stranger.

## 2. Problem statement
- **Churn definition:** customer left the service in the most recent month (`Churn = Yes`).
- **Prediction framing:** snapshot scoring ("as of today, who is likely to be a churner").
  The Telco data has no dates, so this is a simulated scoring run, not a true forecast.
  This limitation is stated in the README.
- **Action triggered:** customers above a score threshold receive a retention offer.

## 3. Goals
- G1. Reproducible baseline: `make data train eval` gives identical numbers from a fresh clone.
- G2. Demonstrate the four sprint skills: Python, SQL, statistics, ML fundamentals.
- G3. Honest evaluation: leakage-safe pipeline, held-out test set used once, bootstrap CIs.
- G4. Decision-oriented: threshold chosen from an explicit cost trade-off, not 0.5.
- G5. Produce an error analysis that seeds the Weeks 3–4 deliverable.

## 4. Non-goals
- No deep learning, no hyperparameter-search marathons, no AutoML.
- No serving, Docker, or monitoring (Weeks 9–10).
- No claims of production readiness or causal conclusions.
- No web UI.

## 5. Users
- **Primary:** the repo author (learning + portfolio).
- **Secondary:** reviewers/recruiters who skim the README and may clone the repo.
- **Fictional business user:** a retention team that acts on the top-scored customers.

## 6. Functional requirements
| ID | Requirement | Acceptance check |
|---|---|---|
| FR-1 | `make data` downloads the dataset and verifies a SHA-256 checksum | Fresh clone: file present, checksum passes |
| FR-2 | Raw CSV is loaded into SQLite (`data/churn.db`) | Table `customers` row count equals CSV rows |
| FR-3 | SQL profiling + segment + cohort queries live in `sql/` and run via code | ≥ 6 queries; outputs reproduced in README |
| FR-4 | Statistics module: Wilson CIs per segment, chi-square tests with effect size | `src/stats.py` + tests on toy data |
| FR-5 | Seeded stratified train/val/test split (60/20/20), IDs saved to `data/splits/` | Same seed gives identical IDs (test) |
| FR-6 | Preprocessing is one sklearn `Pipeline`/`ColumnTransformer` fit on train only | Leakage test passes |
| FR-7 | Models: dummy, logistic regression, gradient boosting | All three in the results table |
| FR-8 | Metrics: PR-AUC, ROC-AUC, recall at precision ≥ 0.5, Brier, confusion matrix at chosen threshold | `src/evaluate.py` unit-tested |
| FR-9 | Threshold chosen on validation by minimizing assumed cost | Cost params in config |
| FR-10 | Calibration plot and error analysis by segment | Notebook 03 |
| FR-11 | Final test evaluation runs once with bootstrap 95% CIs | Writes `reports/final_metrics.json`; refuses to rerun |
| FR-12 | Config-driven runs from `configs/baseline.yaml` | No hardcoded seeds/paths in `src/` |
| FR-13 | CI runs lint + tests on push/PR | Green check on GitHub |
| FR-14 | README: problem, cost framing, results table with CIs, limitations, reproduce steps | Reviewer can reproduce from README alone |

## 7. Non-functional requirements
- **Reproducibility:** pinned `requirements.txt` from `pip freeze`; fixed seeds; documented Python version.
- **Runtime:** full `make train eval` under 5 minutes on a laptop CPU.
- **Quality:** `ruff` clean; tests pass; type hints on public functions.
- **Security/privacy:** per `docs/SECURITY.md`.
- **Documentation:** every module has a docstring; README stays accurate.

## 8. Success metrics
- Beats the majority-class baseline on PR-AUC with non-overlapping 95% CIs (or the README says it does not).
- A third party reproduces the results table from a fresh clone without asking questions.
- Test set evaluated exactly once.
- Not a goal: maximizing accuracy or leaderboard rank.

## 9. Assumptions (to be labeled as assumptions in the README)
- Cost of a false positive ≈ 1 unit (offer cost); cost of a false negative ≈ 5–10 units (lost customer value). These are invented, not measured. Later refinement: derive from `MonthlyCharges`.
- The label reflects the most recent month; features and label overlap in time, which limits any causal reading.

## 10. Risks
| Risk | Likelihood | Mitigation |
|---|---|---|
| Leakage via preprocessing before split | High for beginners | Pipeline-only preprocessing + test |
| Tuning on test set | Medium | Test locked until Day 12; one-shot guard |
| Dataset URL changes or disappears | Medium | SHA-256 check; documented manual fallback |
| Overfitting to a small dataset (~7k rows) | Medium | Stratified CV, CIs, simple models first |
| Scope creep into modeling extras | High | `AGENTS.md` rule 1; roadmap |
| Time-aware dataset (Day 10) too heavy | Medium | Documented fallback: skip and state limitation |

## 11. Milestones
See `docs/ROADMAP.md` (14 days, six phases).

## 12. Open questions
- OQ-1. Use KKBox (time-aware) on Day 10, or document the limitation and skip?
- OQ-2. Exclude `gender` from features? (Default in this PRD: yes, see DATA_CARD.)
- OQ-3. Derive the cost ratio from `MonthlyCharges` or keep fixed assumptions?
