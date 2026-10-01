# Reading the output

The first place to look is `Main_n20`, `Main_n30`, and `Main_n50` in `transition_precision.xlsx`. Each sheet contains all three scenarios and all five baskets, with one row for each originating state.

| Field | Meaning |
|---|---|
| mean_outgoing | Mean number of observed transitions starting from this state |
| empty_probability | Fraction of trials with no observed outgoing transitions from the state |
| row_rmse | Square root of mean squared probability error, averaged over all trials and four destinations |
| nonempty_row_rmse | The same quantity restricted to trials with an observed outgoing row |
| mean_total_variation | Mean of one-half the sum of four absolute probability errors |
| tv_q95 | 95th percentile across trials of the row total-variation error |
| rmse_to_CR / PR / SD / PD | Destination-specific RMSE including empty-row replacement |

`cell_errors.csv` provides the full destination-specific results. `scope=all` is the primary error evaluation. `scope=nonempty` is a conditional diagnostic. The difference in denominators is explicit in `trials_total` and `trials_used`.

`support_strata.csv` shows whether large errors are concentrated at small observed row counts. The rows are retained even when a band has no trials; its estimation-error cells are then missing.

`mse_decomposition.csv` separates the error contributed by uniform replacement in empty rows from error in estimated, nonempty rows. The contributions add to the total MSE, not to total RMSE.

All CSV probabilities and errors are stored as fractions. A row RMSE of 0.22 is an error on the probability scale (22 percentage points), not a relative error. Neither row RMSE nor matrix RMSE is a clinical utility score. No threshold for declaring adequate precision is built into the report.
