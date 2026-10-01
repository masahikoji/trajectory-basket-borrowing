# Software validation summary

These checks concern implementation and reproducibility. They do not establish clinical adequacy or replace inspection of the full manuscript simulation results.

## Core analysis code

- Main tests: 34 passed.
- Assessment-count sensitivity tests: 19 passed.
- Prior replay/source-integrity tests: 28 passed.
- CSV input-interface tests: 8 passed.
- Transition-matrix precision diagnostic tests: 26 passed.
- Total software tests: 115 passed.
- The synthetic CSV example completed through partition selection and posterior inference.
- The packaged Main, follow-up, and prior numerical modules retain the previously verified computational behavior; the source mapping is recorded in `source_map.json`.

## Transition-matrix precision diagnostic

- Reference outputs for all nine Main settings at trial indices 0, 17, and 9999 matched the Main simulation implementation for transition counts, initial/final counts, assessment-count frequencies, response counts, and fitted transition matrices.
- Reports were invariant to tested worker counts and processing-block sizes.
- Saved-input mode and regeneration mode produced identical numerical diagnostic reports for the validation fixture.
- Missing or altered source chunks, incompatible Main definitions, inconsistent fitted matrices, and count-conservation failures are rejected.
- All nine Main settings were evaluated at 10,000 trials per setting for the diagnostic validation run; arithmetic checks included the empty-row/nonempty-row MSE decomposition.

The detailed diagnostic validation record is in `transition_precision/validation/VALIDATION.md`.

## Test environment

Release checks were performed on Linux with Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0, and openpyxl 3.1.5. These records are not a hardware-specific validation of M3 Ultra execution. The transition diagnostic includes a macOS setup helper, but users should confirm the active interpreter and numerical-library versions on their own system.

## Interpretation

The transition diagnostic deliberately does not impose a threshold defining adequate precision. In particular, sparse CR or PR origin rows can remain variable even when the row is nonempty. The reported support and direct probability-estimation errors should be interpreted together.
