# ROADMAP.md — 14-Day Plan (Weeks 1–2 of the ML Engineer sprint)

Rule: do only the day you are told. Each day ends with an exit check.

| Day | Focus | Key files | Exit check |
|---|---|---|---|
| 1 | Setup, deps, data download, problem statement | `Makefile`, `scripts/download_data.py`, `README.md`, `.gitignore`, `requirements.txt` | Fresh clone runs `make data` |
| 2 | Load into SQLite, profile | `src/data.py`, `sql/01_profile.sql` | Profile output in README |
| 3 | Segment + cohort SQL, leakage-suspects list | `sql/02_*.sql`, `sql/03_*.sql`, `src/sqlrun.py` | ≥ 6 queries, 1 finding each |
| 4 | Stats I: Wilson CIs, segment plots | `src/stats.py`, `tests/test_stats.py` | CI plots; can explain wide CIs |
| 5 | Stats II: chi-square + Cramér's V, collinearity, imbalance decision | `notebooks/02_stats.ipynb` | Each test ends with a modeling decision |
| 6 | Seeded stratified split + leakage tests | `src/data.py`, `data/splits/`, `tests/test_split.py` | `pytest` green |
| 7 | Preprocessing pipeline | `src/features.py`, `tests/test_pipeline.py` | Single raw→features path |
| 8 | Baselines (dummy, logreg) + metrics | `src/train.py`, `src/evaluate.py` | 2-row results table |
| 9 | HGB, light CV tuning, run logging | `reports/runs.csv` | 3-model table |
| 10 | Time-aware comparison (or documented skip) | `src/data_kkbox.py` (optional) | README paragraph on random vs time split |
| 11 | Calibration, threshold, error analysis | `notebooks/03_error_analysis.ipynb` | 3 error patterns written up |
| 12 | Final evaluation (once) + bootstrap CIs | `reports/final_metrics.json` | Final table; no retuning |
| 13 | Config, Makefile completion, CI | `configs/`, `.github/workflows/ci.yml` | Cold clone reproduces numbers |
| 14 | README polish, limitations, release | `README.md`, tag `v1.0` | Stranger can reproduce from README |

## Cut order if time runs short
1. Day 10 (time-aware dataset) → keep as a documented limitation.
2. Cohort query on Day 3.
3. Tuning on Day 9.
Never cut: split + leakage tests, pipeline, baselines, one-shot final eval, README results with CIs.
