# HLD — High-Level Design

## 1. Context
Offline batch ML pipeline run locally. No network services. One external dependency:
a public CSV download at setup time.

## 2. Architecture
```
            make data
 [public CSV] ────────► data/raw/telco_churn.csv  (gitignored, checksum-verified)
                               │  src/data.py: load + clean types
                               ▼
                       data/churn.db (SQLite, gitignored)
                         table: customers
                  ┌────────────┴─────────────┐
        sql/*.sql via src/sqlrun.py     src/data.py: split
        (profiling, segments, cohorts)        │
                  │                           ▼
                  │               data/splits/{train,val,test}_ids.csv (committed)
                  ▼                           │
         reports/sql_*.csv                    ▼
                                   src/features.py: ColumnTransformer
                                              │ (fit on train only)
                                              ▼
                                   src/train.py: dummy / logreg / HGB
                                    5-fold stratified CV on train
                                              │
                                              ▼
                              src/evaluate.py: metrics, threshold on val,
                              calibration, bootstrap CIs, final test (once)
                                              │
                                              ▼
                       reports/{metrics.csv, final_metrics.json, figures/}
                                              │
                                              ▼
                                         README results table
```

## 3. Components
| Component | Responsibility | Location |
|---|---|---|
| Data acquisition | Download + checksum | `scripts/download_data.py` |
| Data layer | CSV→SQLite, type fixing, splitting | `src/data.py` |
| SQL analytics | Profiling/segment/cohort queries | `sql/`, `src/sqlrun.py` |
| Statistics | CIs, hypothesis tests, effect sizes | `src/stats.py` |
| Feature pipeline | Imputation, scaling, encoding in one object | `src/features.py` |
| Training | Model definitions, CV, run logging | `src/train.py` |
| Evaluation | Metrics, threshold, calibration, bootstrap, final eval | `src/evaluate.py` |
| Config | Seeds, paths, split sizes, costs, model params | `configs/baseline.yaml`, `src/config.py` |
| Orchestration | One-command workflows | `Makefile` |
| CI | Lint + tests | `.github/workflows/ci.yml` |

## 4. Data flow and boundaries
1. Raw CSV is immutable input. Cleaning happens in code, never by editing the file.
2. Split happens **before** any fitted transformation. Split is by customer ID and persisted.
3. The test partition is read only by `evaluate.py --final`.
4. Models are trained only on train (CV) and train+val (final refit).
5. Threshold is chosen on validation predictions only.

## 5. Key design decisions
| # | Decision | Rationale | Alternative rejected |
|---|---|---|---|
| D1 | SQLite for the SQL layer | Zero setup, file-based, reproducible | Postgres (setup burden, little extra value at 7k rows) |
| D2 | 60/20/20 stratified split with persisted IDs | Validation set for threshold; reproducible | Random split without saved IDs |
| D3 | sklearn Pipeline for all preprocessing | Structural leakage prevention | Manual preprocessing in notebooks |
| D4 | PR-AUC as primary metric | Imbalanced target; ranking quality matters for retention lists | Accuracy |
| D5 | Threshold from cost trade-off on validation | Ties model to the business action | Fixed 0.5 |
| D6 | Class weights for imbalance | Simple, no synthetic data | SMOTE (leakage-prone, unnecessary here) |
| D7 | `HistGradientBoostingClassifier` as the tree model | In sklearn, no extra dependency, handles NaN | XGBoost/LightGBM (extra dependency) |
| D8 | Exclude `customerID` and `gender` from features | ID has no signal; `gender` is a protected attribute with no business justification | Include all columns |
| D9 | One-shot final evaluation guard | Enforces test-set discipline mechanically | Honor system |
| D10 | Config in YAML | Reproducible, reviewable runs | Hardcoded constants |

## 6. Non-functional design
- **Reproducibility:** pinned deps, seeded randomness, committed split IDs, checksum.
- **Observability:** each training run appends params + metrics to `reports/runs.csv`.
- **Failure modes:** download failure (clear exit code), checksum mismatch (hard fail), missing split files (regenerate deterministically), second final-eval attempt (refuse).
- **Performance:** all components fit in memory (~7k rows); CPU only.

## 7. Out of scope (later sprint weeks)
Experiment tracking server, model registry, serving API, Docker, monitoring.

## 8. Extension point (Day 10)
A second dataset adapter (`src/data_kkbox.py`) producing the same interface
(`load() -> DataFrame`, `time_split(df, cutoff) -> (train, test)`) to compare random vs time-based splits. Optional.
