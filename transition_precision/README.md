# Transition-matrix precision and stability in the Main simulations

This diagnostic evaluates estimation error of the transition matrices in Main Scenarios 1-3 at 20, 30, and 50 patients per basket. It does not change the proposed method, fit a new transition model, select clusters, or recompute ORR posteriors.

## Setup on an Apple Silicon Mac

From the `transition_precision` directory:

```bash
bash setup_mac.sh
```

The script locates a native Python 3.11-3.13 interpreter and creates a separate local environment at `~/.venvs/trajectory_basket_transition_precision`. It does not modify the earlier simulation environments. If automatic discovery fails, specify an installed interpreter:

```bash
PYTHON=/absolute/path/to/python3.13 bash setup_mac.sh
```

Subsequent commands use an absolute interpreter path; activating Conda or a different virtual environment is unnecessary.

## Main analysis: no previous results folder is required

```bash
PY="$HOME/.venvs/trajectory_basket_transition_precision/bin/python"
"$PY" test_suite.py
"$PY" evaluate.py --plan-only
"$PY" evaluate.py --smoke --workers 20 --out "$HOME/basket_runs/smoke_transition_precision"
```

The plan contains 9 settings. The smoke check uses 20 trials per setting and must not be used for inference.

Production:

```bash
caffeinate -i "$PY" evaluate.py --reps 10000 --workers 20 --out "$HOME/basket_runs/results_transition_precision"
"$PY" report.py --results "$HOME/basket_runs/results_transition_precision"
```

The first command generates observed transition counts and exports checked CSV summaries and a LaTeX table. The second creates the Excel workbook from those CSVs. No simulation is repeated by `report.py`.

Alternatively, after setup:

```bash
bash run_main_m3.sh
```

The runner performs the tests, the 10,000-trial analysis, and Excel export. `WORKERS`, `REPS`, and `OUT` can be specified as environment variables. Defaults are 20 workers, 10,000 trials per setting, and the local output path above.

## Optional: use the exact saved trials instead

If a complete `results_all` from a compatible completed Main simulation result directory is available:

```bash
"$PY" evaluate.py --input /absolute/path/to/results_all --inspect
caffeinate -i "$PY" evaluate.py --input /absolute/path/to/results_all --reps 10000 --workers 20 --out "$HOME/basket_runs/results_transition_saved"
"$PY" report.py --results "$HOME/basket_runs/results_transition_saved"
```

This mode uses `transition_counts`, `initial_counts`, `final_counts`, `length_counts`, `responses`, and `Phat` from the selected Main raw files. It verifies that saved `Phat` equals the row-count estimator and checks state-flow conservation. Source files are hashed and remain read-only. Extra supplementary, nonhomogeneous, and stress settings are not analyzed. A missing requested chunk stops the analysis; partial data are never silently substituted or used.

Multiple completed input directories are accepted after `--input` if they contain nonoverlapping Main settings. A duplicate setting is rejected. An aggregate Excel workbook cannot replace the raw count files.

## Reproducibility

Regeneration uses the original Main setting names, per-trial SeedSequence, basket permutation, and patient-uniform ordering. The default seed is 610202610. With the matching generating implementation and random-number behavior, counts coincide with the original Main trials. Reference fixtures cover all nine settings. Changing the seed generates a new Monte Carlo evaluation, not a replay of the original exact dataset. The saved-input mode is preferable when exact original count-file identity is required.

Resume an interrupted run using the identical arguments and `--resume`. The number of workers may change; scientific settings, chunk size, input hashes, source code, and NumPy/Python versions may not. Do not remove a running process's `.running.lock`. After an abnormal termination, confirm that the old process has stopped before removing a stale lock.

## Outputs

- `transition_precision.xlsx`: sample-size summaries and detailed error/support sheets.
- `cell_errors.csv`: all 16 probabilities, separately for all trials and nonempty rows. Bias, empirical SD, MSE, RMSE, MAE, Monte Carlo SEs, quantiles, and absolute-error tail probabilities.
- `row_errors.csv`: support and empty-row diagnostics, unconditional and nonempty-row RMSE/TV, and destination-specific RMSE.
- `support_strata.csv`: conditional row errors for outgoing counts 0, 1-4, 5-9, 10-19, and 20+.
- `mse_decomposition.csv`: exact decomposition of total MSE into empty-row and nonempty-row contributions.
- `observation_counts.csv`, `matrix_errors.csv`, `true_transitions.csv`: observation patterns, overall matrix error, and generating probabilities.
- `transition_precision_table.tex`: row-level LaTeX table; requires `booktabs` and `longtable`.
- `raw/*.npz`, `manifest.json`, `execution.json`, `checks.json`: retained trial counts, software/settings provenance, completion status, and arithmetic checks.

Probability estimates and their errors are in probability units: RMSE 0.10 means 10 percentage points, not 0.10 percent. The empirical 5th/95th quantiles describe Monte Carlo variability and are not confidence intervals for the estimator's mean. No clinical adequacy cutoff is imposed.

## Scope

Only the original homogeneous Main study and its primary assessment-count distribution are included. The program does not independently vary the horizon law, introduce informative dropout, change the uniform empty-row convention, or certify downstream borrowing performance. Different Main mechanisms and states produce different state occupancy; the support-stratified tables are descriptive, not randomized comparisons of different exposure amounts.

See `METHODS.md` and `validation/VALIDATION.md`.
