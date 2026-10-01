# Trajectory-informed information borrowing in basket trials

Python implementation accompanying **A Trajectory-Informed Clustering Approach for Information Borrowing in Basket Trials**.

The repository contains the proposed trajectory-informed clustering method, conditional hierarchical ORR inference, all simulation and sensitivity analyses reported with the study, and the additional transition-matrix precision diagnostic used to evaluate estimation stability in the Main simulations.

## Scope

The implemented analysis uses five baskets with equal patient counts. Each patient has one to ten consecutive categorical response assessments with states CR, PR, SD, or PD. The software does not impute intermittent missing assessments, model informative dropout, or support unequal basket sizes in the current implementation. The example dataset is synthetic.

## Installation

Use Python 3.11-3.13. On Apple Silicon, use an `arm64` interpreter. From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python tools/check_environment.py
```

Python 3.12 or 3.13 can be used instead of 3.11.

## Analyze one dataset

```bash
python examples/analyze_trial.py \
  --input examples/synthetic_trial.csv \
  --p0 0.30 \
  --out results/example
```

The input columns are `basket`, `patient`, `assessment`, and `state`. Assessment numbers must be consecutive within patient and start at one. `--p0` is a prespecified null response rate. See `examples/README.md`.

## Reproduce the manuscript simulations

### Main, lower-response Supplement, time-nonhomogeneity, stress, and mixture-weight analyses

```bash
python main/simulate.py \
  --suite all \
  --include-stress \
  --weight-sensitivity \
  --plan-only

python main/simulate.py \
  --suite all \
  --include-stress \
  --weight-sensitivity \
  --reps 10000 \
  --workers 8 \
  --cache .cache/main \
  --out results/main
```

This produces 34 generating settings: 9 Main, 3 lower-response Supplement, 18 time-nonhomogeneous, and 4 additional stress settings. Mixture weights 0.25 and 0.75 are applied to the same generated trials and do not add generating settings.

### Assessment-count distribution sensitivity

```bash
python followup/simulate.py \
  --profiles short long \
  --sizes 20 30 50 \
  --scenarios 1 2 3 \
  --reps 10000 \
  --workers 8 \
  --cache .cache/followup \
  --out results/followup
```

This adds 18 generating settings: Short and Long assessment-count distributions crossed with Main Scenarios 1-3 and n=20,30,50.

### Precision-prior sensitivity

The prior-scale analysis reuses the completed Main and lower-response trials. It does not generate new patient trajectories.

```bash
python prior/reanalyze.py --input results/main --inspect
python prior/reanalyze.py \
  --input results/main \
  --workers 6 \
  --cache .cache/prior \
  --out results/prior
```

The source result directory must contain its `manifest.json` and all required `raw/*.npz` files.

### Transition-matrix precision and stability diagnostic

This diagnostic addresses estimation precision of the fitted 4-state transition matrices in the nine Main settings. It reports row support, empty-row frequency, cell-level bias/RMSE/MAE, row-level error, support-stratified error, and the contribution of empty-row replacement to MSE.

```bash
python transition_precision/evaluate.py --plan-only
python transition_precision/evaluate.py \
  --reps 10000 \
  --workers 8 \
  --out results/transition_precision
python transition_precision/report.py \
  --results results/transition_precision
```

It can also read the exact saved Main trial counts from a completed `results/main` directory:

```bash
python transition_precision/evaluate.py \
  --input results/main \
  --inspect
```

The diagnostic does not alter clustering, refit a different transition model, or recompute the hierarchical ORR posterior. Its definitions and output schema are documented in `transition_precision/METHODS.md` and `transition_precision/DATA_DICTIONARY.md`.

## Analysis inventory

The complete simulation study contains 52 distinct generating settings:

- 9 Main settings;
- 3 lower-response Supplement settings;
- 18 time-nonhomogeneous settings;
- 4 additional stress settings;
- 18 assessment-count sensitivity settings.

Prior-scale and mixture-weight analyses reuse generated trials rather than adding generating settings. The transition-matrix precision diagnostic evaluates the nine Main settings and likewise does not add a new data-generating mechanism. See `docs/ANALYSIS_MAP.md` for the manuscript-to-code mapping.

## Verification

Run all software tests:

```bash
python tools/run_tests.py
```

Numerical posterior checks are available separately:

```bash
python main/validate_numerics.py --cache .cache/numerical --out results/numerical
python prior/validate_priors.py --help
```

Validation records for the distributed code are summarized in `docs/VALIDATION.md`. Software tests and small smoke runs are not substitutes for the 10,000-replicate manuscript analyses.

## Statistical interpretation

`Proposed` uses an occupancy/ORR partition criterion followed, only when two groups are selected, by a cluster-count-preserving membership refinement using fitted response-attainment and ordinal trajectory summaries. Only observed binary response counts enter the ORR likelihood.

The hierarchical posterior integrates cluster-level hyperparameters conditional on the selected partition. Partition-selection uncertainty is not integrated. The equal-tail 90% credible-interval decision rule is not calibrated to guarantee a 5% frequentist error rate.

## Citation and archival release

Software citation metadata are provided in `CITATION.cff`. For a published release, cite the version-specific DOI assigned by Zenodo to the archived GitHub release.
