# TESTING.md — Test Strategy

Run: `make test` (`pytest -q`). All tests use small synthetic DataFrames; none require the real dataset
except `tests/test_data_integration.py`, which is skipped if `data/raw/telco_churn.csv` is absent.

## Required tests
| File | Test | Asserts |
|---|---|---|
| `tests/test_data.py` | `test_load_raw_schema` | Expected columns present; `TotalCharges` numeric; `churn` in {0,1} |
| | `test_unique_ids` | Duplicate `customerID` raises |
| `tests/test_split.py` | `test_no_overlap` | Train/val/test ID sets pairwise disjoint |
| | `test_covers_all_rows` | Union of splits equals all IDs |
| | `test_stratification` | Churn rate in each split within ±1.5 percentage points of overall |
| | `test_deterministic` | Same seed → identical IDs; different seed → different IDs |
| | `test_load_split_detects_overlap` | Tampered ID files raise |
| `tests/test_pipeline.py` | `test_scaler_fit_on_train_only` | After fit on train, scaler mean equals train mean, not train+val mean |
| | `test_unseen_category_ok` | Unseen category at transform time does not crash |
| | `test_dropped_columns_absent` | `customerID`, `gender` not in transformed feature names |
| | `test_no_target_in_features` | `churn`/`Churn` absent from features |
| `tests/test_stats.py` | `test_wilson_known_case` | Matches a hand-computed interval within tolerance |
| | `test_cramers_v_bounds` | 0 ≤ V ≤ 1; independent data gives V ≈ 0 |
| | `test_bootstrap_ci_contains_point` | low ≤ point ≤ high; seed gives identical output |
| `tests/test_evaluate.py` | `test_pr_auc_perfect_and_random` | 1.0 for perfect scores; ≈ positive rate for constant scores |
| | `test_recall_at_precision` | Hand-built example returns known value; 0.0 when floor unreachable |
| | `test_choose_threshold_cost` | Raising `c_fn` lowers (or keeps) the chosen threshold |
| | `test_confusion_counts_sum` | tn+fp+fn+tp == n |
| | `test_final_eval_one_shot` | Second call without `force` refuses |
| `tests/test_config.py` | `test_split_fractions_sum_to_one` | train+val+test == 1.0 |

## Reproducibility check (manual, Day 13–14)
Fresh clone → venv → `pip install -r requirements.txt` → `make data db sql split train eval`.
Compare results table to the README. Differences beyond floating-point noise are bugs.

## Leakage checklist (review before Day 12)
- [ ] No `fit`/`fit_transform` outside a Pipeline trained on training rows
- [ ] No feature derived from the target or from `TotalCharges` computed using future info without justification
- [ ] Test IDs never loaded before `--final`
- [ ] Threshold chosen on validation only
- [ ] No hyperparameter chosen by looking at test metrics

## Coverage target
Core logic (`data`, `features`, `stats`, `evaluate`) ≥ 80% line coverage. Do not write tests that only restate the implementation.
