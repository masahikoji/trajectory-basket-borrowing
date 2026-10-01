# Statistical implementation

## Input and target

The analysis uses five baskets with a common patient count n and categorical states ordered CR, PR, SD, PD. A patient is a responder if any observed assessment is CR or PR. Every patient's scheduled observations are retained after first response. Consecutive assessment indices represent discrete steps; missing visits and informative discontinuation are not modeled.

For basket j, the initial distribution is the empirical distribution of first assessments. Each transition-matrix row is estimated from all observed outgoing transitions in that basket. A row with no outgoing observations is assigned probabilities (1/4,1/4,1/4,1/4). The empirical distribution of assessment counts is w_jt.

## Occupancy and partition selection

The fitted mean-state occupancy distribution is

    o_j = sum_t w_jt (1/t) sum_(l=0)^(t-1) pi_j P_j^l.

Its ordinal score is s_j = o_j . (1,2/3,1/3,0). The initial two-vector is (s_j, observed_ORR_j). For each component, delete one whole patient and refit the feature. Jackknife pseudovalues are

    u_ji = n theta_j - (n-1) theta_j,-i.

The common within-basket covariance estimate is

    S = sum_j sum_i (u_ji - mean_i u_ji)(u_ji - mean_i u_ji)' / [5(n-1)].

Let z_j = S^(-1/2)(theta_j - mean_j theta_j) and, for a partition C,

    B_C = n sum_c |c| mean_c(z) mean_c(z)'.

All 52 partitions of five baskets are compared. For the two-dimensional working model, with eigenvalues lambda_1 <= lambda_2 of B_C, the criterion is

    G_C = (lambda_1+lambda_2)/2 + 2 log I_0((lambda_2-lambda_1)/4),
    penalty_C = sum_c log(n |c|) - log(5n),
    criterion_C = G_C - penalty_C.

Covariance rank reduction and numerical ties are handled in `cluster_selection.py`. This criterion is a working information criterion, not an exact posterior probability on partitions.

## Two-group membership refinement

The selected number of clusters K is retained. If K is not two, membership is unchanged. If K equals two, all 15 two-block partitions are compared without imposing block sizes.

Let N={SD,PD} and Q_j be the corresponding submatrix of the fitted P_j. The second-stage clustering feature is

    h_j = 1 - sum_t w_jt pi_j,N Q_j^(t-1) 1.

This estimates ever-response under the fitted working model. It does not replace the observed responder count in the ORR likelihood. Refit (s_j,h_j) and its joint patient-jackknife covariance. The membership criterion combines the orientation-averaged contrast G_C with

    A_C = n sum_c |c| (mean_c(s)-mean_j(s))^2 / S_11,
    refined_C = 2 log[(1-w) exp(G_C/2) + w exp(A_C/2)] - penalty_C.

The primary weight is w=.5; .25 and .75 are optional sensitivity settings. A numerical tie retains initial membership when it is a maximizer; otherwise the fixed randomized tie rule applies. Keeping K fixes cluster-count accuracy, but does not guarantee the quality of borrowing partners or downstream efficacy inference.

## Conditional ORR model

For the selected cluster c:

    r_j | eta_j ~ Binomial(n, expit(eta_j))
    eta_j | mu_c,tau_c ~ Normal(mu_c, 1/tau_c)
    mu_c ~ Normal(0,1)
    tau_c ~ Gamma(shape=2, rate=1).

The numerical posterior calculation integrates mu_c and tau_c. It remains conditional on the selected partition and does not account for uncertainty in partition selection. The independent comparator uses Beta(1+r_j,1+n-r_j). EXNEX uses EX prior probability .5, the same EX hierarchy, and an independent N(0,1) NEX log-odds distribution.

A basket is declared active if the lower endpoint of its equal-tail 90% posterior interval exceeds p0. This is not a general 5% frequentist error guarantee. Gamma-rate sensitivity changes rate to .5 and 2 with shape fixed, so the prior means of within-cluster variance are .5,1,2.

## Sensitivity scope

Time-inhomogeneity changes early/late data generation while retaining a homogeneous fitted analysis model. Follow-up sensitivity changes the noninformative distribution of the number of observations. Prior sensitivity reuses saved counts and partitions. None of these, alone or together, establish robustness to informative dropout or unobserved response assessments.

For manuscript simulation truths, the ever-response ORR is computed from the generating nonresponse submatrices and the assessment-count distribution. Time-varying settings use ordered products, not powers of an averaged matrix. Generating probabilities are not passed to the observed-data clustering functions.
