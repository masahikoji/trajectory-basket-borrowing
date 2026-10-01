# Observed-trajectory example

`synthetic_trial.csv` is a synthetic demonstration with five baskets and 20 patients per basket. It is not a clinical dataset or a set of manuscript simulation results.

CSV fields:

| Field | Definition |
|---|---|
| basket | Nonempty basket identifier |
| patient | Patient identifier, unique within its basket |
| assessment | Consecutive observed index 1,...,T, with 1 <= T <= 10 |
| state | CR, PR, SD, or PD |

Each basket must contain the same number of patients (at least three). Identifiers are sorted, with a recorded random permutation used before partition selection. Clinical efficacy is assessed by ever attaining CR/PR among the observed assessments. The input must contain the complete observations used in that definition, including observations after response.

The wrapper rejects missing assessment indices, duplicate records, missing states, unequal basket sizes, or more than five baskets. It does not interpret NA as PD or join states across unobserved visits. Actual elapsed time is not an input; equal discrete steps remain a modeling assumption.

From the repository root:

```bash
python examples/analyze_trial.py --input examples/synthetic_trial.csv --p0 0.30 --out results/example
```

The example uses an illustrative p0 of .30. For an application, specify the design's null response threshold before examining these data; do not substitute the dataset's fitted response probability.

Outputs are `basket_summary.csv` and `analysis.json`. The first records observed response counts, selected groups, conditional posterior summaries, and active-basket decisions. The second records the input hash, numerical versions, seed, prior, and threshold. Keep both for a reproducible analysis. Running with the same input and seed reproduces partition tie handling. Conditional posterior intervals are not uncertainty intervals for the selection procedure as a whole.
