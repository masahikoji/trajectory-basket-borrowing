# Transition estimation diagnostics

## Settings and estimator

Nine settings are evaluated: Main Scenarios 1-3 at n=20,30,50 patients in each of five baskets. The original initial distributions, time-homogeneous transition probabilities, and assessment-count law are retained. The assessment-count probabilities for T=1,...,10 are

(0.03, 0.05, 0.25, 0.25, 0.20, 0.10, 0.05, 0.03, 0.02, 0.02),

with mean 4.45. All observations up to T are retained, including observations after initial CR/PR. The first assessment is not a pretreatment state.

For each trial b, basket j, origin r, and destination s, let C[b,j,r,s] be the observed transition count and N[b,j,r] its row sum. The estimator is

P_hat[b,j,r,s] = C[b,j,r,s] / N[b,j,r] when N>0;
P_hat[b,j,r,s] = 0.25 when N=0.

The estimator accepts observed counts only. Generating probabilities enter the error evaluation, never the estimator. No smoothing or prior is added to this estimator.

## Cell-level error

For d_b=P_hat[b,j,r,s]-P_true[j,r,s], report

- Bias = mean_b d_b.
- Empirical SD = sample SD of P_hat across trials.
- MSE = mean_b d_b^2; RMSE = sqrt(MSE).
- MAE = mean_b |d_b|.
- MCSE(Bias) = sample SD(d_b)/sqrt(R).
- MCSE(MSE) = sample SD(d_b^2)/sqrt(R).
- MCSE(RMSE) = MCSE(MSE)/(2 RMSE), a delta-method approximation for positive RMSE.

The 5th percentile, median, and 95th percentile of P_hat, 95th percentile of absolute error, and proportions with absolute error greater than 0.10 and 0.20 are also reported. These thresholds are descriptive probability-scale errors, not justified clinical acceptability bounds. Relative errors are not reported because some generating probabilities are zero.

The primary calculation includes ALL R trials and thus the uniform replacement in empty rows. A second calculation conditions on N>0 and uses the actual number R_plus of such trials. Empty conditional subsets are reported as missing, not zero. Empirical SD is not the standard error within a particular trial. MCSE concerns the uncertainty of a simulation summary.

## Row-level error

Define row MSE as mean_b [(1/4) sum_s d_bs^2], and row RMSE as its square root. This is an equal-destination probability error, not a relative error or a count-weighted norm. It is not the mean of four cell RMSEs and is not the average of per-trial row RMSEs.

Also report TV_b=(1/2)sum_s |d_bs|, mean(TV), Q95(TV), mean(max_s |d_bs|), and Q95(max_s |d_bs|). The overall matrix RMSE similarly averages squared error over all 16 entries; it is secondary because an overall average can hide a weakly informed CR row.

All rows and all baskets remain separate. Baskets following the same generating mechanism are not treated as extra patients or combined into a transition estimator.

## Support and empty-row contribution

For each origin, report mean outgoing count, its empirical 5th/50th/95th quantiles, empty-row frequency, and frequencies below 5, 10, and 20 outgoing transitions. Total visits in a state equal outgoing transitions from that state plus the final-state count. The corresponding observed state fraction is visit-weighted and is labeled as such; it is not the patient-averaged mean-state-occupancy feature used for clustering.

For each cell,

MSE_all = f_empty*(0.25-P_true)^2 + (1-f_empty)*MSE_nonempty.

When every row is empty, the second contribution is zero even though the conditional MSE is undefined. The exported decomposition is checked numerically. This quantifies how much of the observed error comes from the empty-row convention rather than simply deleting those cases.

Outgoing counts are also stratified as 0, 1-4, 5-9, 10-19, and 20+. These are diagnostic conditional summaries, not prespecified eligibility rules or a causal effect of increasing support. Because row support is partly determined by the realized trajectory, conditional groups need not have otherwise comparable distributions.

## Analytic support check

The expected outgoing count for state r is

n * sum_t w_t * sum_(ell=0)^(t-2) [pi P^ell]_r.

For a single patient, the probability of no outgoing transition from r equals 1 at T=1, and pi[-r] P[-r,-r]^(t-2) 1 at T=t>=2. Average this probability over w_t and raise it to n to obtain the basket empty-row probability. The simulation estimates are exported next to these exact values as an additional diagnostic.

## Independence and interpretation

Precision is evaluated across independently simulated whole trials. No calculation treats repeated transitions within a patient as independent patients, and no naive conditional multinomial standard-error approximation is used. The 4-by-4 stochastic matrix has at most 12 free parameters, but accuracy is reported for all 16 probabilities, including structural zeros.

This analysis measures transition-estimation accuracy in the Main settings. It does not establish that every row has adequate precision, that the homogeneous working model is correct in clinical practice, or that a given row error translates directly into an error rate for the downstream ORR analysis. The horizon distribution is not independently varied in this Main-only analysis.

## Reproduction and execution

The regeneration stream is SeedSequence([seed, stable_id(setting_name), replicate, 0]), followed by the original basket permutation and a 5-by-n-by-11 uniform array. Stable_id uses the first four SHA-256 bytes in little-endian order. Generation and estimation are separate functions. Parallel execution explicitly uses the spawn context; worker count and task completion order do not enter the random seed.

Implementation references:
- Python multiprocessing: https://docs.python.org/3.11/library/multiprocessing.html
- NumPy parallel random generation: https://numpy.org/doc/stable/reference/random/parallel.html

These references concern implementation, not evidence that the transition estimator is sufficiently precise.
