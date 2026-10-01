# Manuscript-to-code analysis map

| Analysis | Code | Generating settings |
|---|---|---:|
| Main Scenarios 1-3, n=20,30,50 | `main/simulate.py --suite main` | 9 |
| Lower-response Supplement, n=30 | `main/simulate.py --suite supplement` | 3 |
| Time-nonhomogeneous sensitivity | `main/simulate.py --suite sensitivity --nonhom-only` | 18 |
| Additional stress configurations | `main/simulate.py --include-stress` | 4 |
| Mixture-weight 0.25/0.50/0.75 | `main/simulate.py --weight-sensitivity` | reuses trials |
| Unrefined/component comparisons | default methods in `main/simulate.py` | reuses trials |
| ORR-only, One-cluster, Independent, EXNEX | default methods in `main/simulate.py` | reuses trials |
| Assessment-count Short/Long sensitivity | `followup/simulate.py` | 18 |
| Precision-prior Gamma(2,0.5/1/2) sensitivity | `prior/reanalyze.py` | replays 12 settings |
| Transition-matrix precision/stability | `transition_precision/evaluate.py` | evaluates Main 9 |

The 52 distinct data-generating settings are 9 Main + 3 lower-response + 18 time-nonhomogeneous + 4 stress + 18 assessment-count settings. The remaining analyses reuse these generated trials or evaluate estimators within them.

The transition-matrix diagnostic reports direct estimation error for `P_hat`: cell-level bias, RMSE and MAE; row-level RMSE and total-variation error; observed row support and empty-row frequency; errors conditional on support bands; and decomposition of MSE into empty-row replacement and nonempty-row estimation components.
