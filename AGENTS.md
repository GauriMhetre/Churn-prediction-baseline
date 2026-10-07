# AGENTS.md — Instructions for AI coding assistants

Read this file first, then `docs/PRD.md`, `docs/HLD.md`, `docs/LLD.md`,
`docs/SECURITY.md`, `docs/TESTING.md`, `docs/DATA_CARD.md`, `docs/ROADMAP.md`.
If these documents conflict, precedence is: SECURITY > PRD > HLD > LLD > ROADMAP.

## Project in one paragraph
`Churn-prediction-baseline` is a 14-day, reproducible baseline for customer churn
prediction on the Telco Customer Churn dataset. Deliverable: a repo anyone can clone and
run with `make data train eval` to get the same numbers. It must demonstrate Python,
SQL, statistics, and ML fundamentals, with leakage-safe preprocessing and honest evaluation.

## Hard rules (never violate)
1. **Work only on the day/task the user names** (see `docs/ROADMAP.md`). Do not build ahead.
2. **Never touch the test set** except in `src/evaluate.py --final`, once, on Day 12.
   Do not tune, plot, inspect, or compute any statistic on test rows before then.
3. **No leakage.** Every imputer, scaler, and encoder is fit on training data only,
   inside an sklearn `Pipeline`. Never call `fit`/`fit_transform` on full data.
4. **No hand-typed dependency versions.** Versions in `requirements.txt` come from
   `pip freeze` output only.
5. **No secrets in the repo.** No tokens, keys, or `.env` contents committed. See `docs/SECURITY.md`.
6. **Never** run `git push --force`, rewrite history, or delete branches.
7. **Never commit** `data/raw/*` (except `.gitkeep`), `data/*.db`, `.venv/`, or notebooks with large outputs.
8. **If a command fails, show the exact error and stop.** Do not work around it or invent a fix silently.
9. **Do not invent results.** Every number in the README must come from a command you ran.
   If you did not run it, say so.
10. **Ask before** adding a dependency, changing the split, changing the metric set, or changing the cost assumptions.

## Conventions
- Python 3.10+; type hints on public functions; docstrings on modules and public functions.
- Formatting/lint: `ruff`. Must pass `make lint` before every commit.
- Tests: `pytest`. Must pass `make test` before every commit.
- All randomness uses the seed from `configs/baseline.yaml` (default 42).
- All paths come from config or `pathlib`; no absolute paths.
- SQL goes in `sql/*.sql` and is executed by code in `src/`; no ad hoc SQL in notebooks only.
- Notebooks are for exploration; any logic needed for results lives in `src/`.
- Commits: small, one logical change, imperative message ("Add stratified split and leakage test").

## Definition of done for any task
- Code + tests written, `make lint` and `make test` pass.
- Relevant doc updated if behavior changed (README results table, LLD signature, etc.).
- Report: files changed, commands run with real outputs, anything uncertain.

## Final report format (end of every task)
```
FILES CHANGED: ...
COMMANDS RUN + REAL OUTPUT: ...
TESTS: pass/fail counts
DEVIATIONS FROM DOCS: ...
UNSURE ABOUT: ...
```
