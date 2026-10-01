# Validation

## Software and reproduction checks

- 26 unit/input/statistical-definition tests passed (`tests.log`).
- The Main simulation implementation supplied reference outputs for all nine Main settings at trial indices 0, 17, and 9999: 27 trials. Six observed-data arrays per trial, including transition counts and Phat, match exactly (`reference_trials.npz`, `reference_sources.json`).
- Reports were identical across 1, 4, and 20 workers with different processing-block sizes. Individual trial seeds do not depend on worker count or scheduling.
- Saved-input mode produced the same seven numerical CSV reports as regeneration for a 9-setting, 20-trial schema fixture.
- Missing chunks, modified input files, changed Main distributions, duplicate source settings, inconsistent Phat, and broken count conservation are rejected.
- An intentionally removed checkpoint was regenerated on resume and the numerical reports remained identical.

## Full Main diagnostic

All nine settings were evaluated at 10,000 trials per setting (90,000 simulated trials). Generation, count validation, CSV summaries, MSE decomposition, and LaTeX output completed with 20 observed worker processes. Excel was then generated in a separate report command. See `main_execution.json` and `main_checks.json`.

The run used Linux x86_64, Python 3.13.5, NumPy 2.3.5, and openpyxl 3.1.5. It is not an M3 Ultra hardware test and is not a rerun of the hierarchical ORR posterior. The saved-input fixture is not the user's full production results directory.

## Spreadsheet checks

Excel formula caches were checked against the numerical summaries. Independent spreadsheet recalculation agreed for 60,993 numerical cells within 1e-10 relative-to-unit scale. Representative ranges from all ten sheets were rendered and inspected. Blank conditional summaries remain blank when no eligible trials exist.

## Interpretation

These checks establish implementation and arithmetic consistency in the tested cases, not adequate clinical precision of every transition row. In the reference-A CR row, limited outgoing transitions continue to produce substantial estimation variability even when the row is nonempty. The software reports this rather than imposing or asserting a passing accuracy threshold.
