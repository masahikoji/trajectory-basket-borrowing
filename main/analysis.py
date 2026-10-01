"""Observed-trajectory clustering and conditional hierarchical ORR inference."""
from __future__ import annotations
import numpy as np
from cluster_selection import PatientData, ORDINAL, information_partition, fitted_distributions
from refinement import analyze as frozen_refinement, METHODS as FROZEN_METHODS, fitted_hitting_probability
from reference.core import estimated_features, choose_partition, canonical_labels
from posterior import FIELDS, independent_table

DEFAULT_METHODS = ('Proposed', 'Previous', 'ORR_only', 'One_cluster', 'Independent',
                   'EXNEX', 'Corrected_original', 'Empirical_occupancy')
WEIGHT_METHODS = ('Refined_w25', 'Refined_w75')
ALL_METHODS = DEFAULT_METHODS + WEIGHT_METHODS + ('Refined_uniform', 'Refined_raw50', 'Empirical_final')
METHOD_MAP = {'Proposed': 'Hitting_mix50', 'Previous': 'Baseline',
              'Refined_w25': 'Hitting_mix25', 'Refined_w75': 'Hitting_mix75',
              'Refined_uniform': 'Hitting_uniform', 'Refined_raw50': 'Raw_mix50'}
METHOD_DESCRIPTIONS = {
    'Proposed': 'Frozen Occupancy_ORR_IC K selection, K=2 membership refinement Hitting_mix50',
    'Previous': 'Frozen Occupancy_ORR_IC before K=2 membership refinement',
    'ORR_only': 'Observed ever-response proportion; original L1/average-linkage/silhouette rule',
    'One_cluster': 'All baskets in one logistic-normal hierarchical group; partial pooling, not common p',
    'Independent': 'Independent Beta(1,1) analyses of actual response counts; no borrowing',
    'EXNEX': 'EXNEX model specified in manuscript; all latent EX indicators and mu/tau integrated',
    'Corrected_original': 'Estimated weighted final-state probabilities plus observed ORR; original silhouette',
    'Empirical_occupancy': 'Empirical patient-averaged ordinal occupancy plus observed ORR; same baseline IC',
    'Refined_w25': 'K-preserving fitted-hitting refinement, directional mixture weight 0.25',
    'Refined_w75': 'K-preserving fitted-hitting refinement, directional mixture weight 0.75',
    'Refined_uniform': 'K-preserving fitted-hitting refinement, uniform direction only',
    'Refined_raw50': 'K-preserving refinement using observed ORR, directional mixture weight 0.50',
    'Empirical_final': 'Empirical last-observed state distribution plus ORR; original silhouette',
}


def validate_observed(data):
    data.validate()
    if data.lengths.shape != (5, data.n, 10):
        raise ValueError('This frozen implementation expects ten possible observation counts.')
    for x in (data.initial, data.transitions, data.lengths, data.final, data.response):
        if not np.all(x == np.floor(x)):
            raise ValueError('Observed counts and indicators must be integers.')
    if np.any(data.response > 1):
        raise ValueError('One binary ever-response indicator per patient is required.')
    lengths = data.lengths @ np.arange(1, 11)
    if not np.array_equal(data.transitions.sum((-2, -1)), lengths-1):
        raise ValueError('Transition totals disagree with patient follow-up lengths.')
    flow = data.initial + data.transitions.sum(-2) - data.transitions.sum(-1)
    if not np.array_equal(flow, data.final):
        raise ValueError('First state, transitions and final state are inconsistent.')
    visits = data.initial + data.transitions.sum(-2)
    if not np.allclose(data.occupancy * lengths[..., None], visits, atol=1e-12, rtol=0):
        raise ValueError('Occupancy does not match observed trajectories.')
    if not np.array_equal(data.response, (visits[..., :2].sum(-1) > 0).astype(int)):
        raise ValueError('Response indicator does not match ever CR/PR in the trajectory.')


def select_partitions(data: PatientData, rng: np.random.Generator,
                      comparator_rng: np.random.Generator):
    """Select all prespecified methods with fixed RNG ordering.

    The frozen function is invoked without any alteration. Enabling/disabling
    output methods therefore cannot change Proposed or Previous selections.
    """
    validate_observed(data)
    frozen, _ = frozen_refinement(data, rng)
    choices = {new: frozen[FROZEN_METHODS.index(old)].copy() for new, old in METHOD_MAP.items()}
    agg = data.aggregate()
    X, Phat, outgoing = estimated_features(agg)
    choices['Corrected_original'] = choose_partition(X)[0]
    choices['ORR_only'] = choose_partition(X[:, -1:])[0]
    choices['Empirical_final'] = choose_partition(np.column_stack((agg.final_counts/data.n,
                                                                 agg.response_counts/data.n)))[0]
    empirical = data.occupancy @ ORDINAL
    means = np.column_stack((empirical.mean(1), data.response.mean(1)))
    pseudo = np.stack((empirical, data.response), axis=-1)
    choices['Empirical_occupancy'] = information_partition(means, pseudo, data.n, comparator_rng)
    choices['One_cluster'] = np.zeros(5, int)
    choices['Independent'] = np.arange(5)
    choices['EXNEX'] = np.full(5, -1, dtype=int)  # No hard EXNEX partition.
    final, occupancy = fitted_distributions(agg.initial_counts, agg.transition_counts, agg.length_counts)
    hit = fitted_hitting_probability(agg.initial_counts, agg.transition_counts, agg.length_counts)
    diag = dict(final_feature=final, occupancy_feature=occupancy, fitted_hitting=hit,
                Phat=Phat, outgoing=outgoing)
    if any(len(np.unique(choices[x])) != len(np.unique(choices['Previous'])) for x in METHOD_MAP):
        raise AssertionError('The frozen K-preserving property has been violated.')
    return choices, diag


def infer_partitions(response_counts, n, choices, methods, p0, legacy_p0, engine):
    """Bayesian ORR inference conditional on observed-data selected partitions."""
    response_counts = np.asarray(response_counts, dtype=int)
    out = np.empty((len(methods), 5, len(FIELDS)))
    labels = np.empty((len(methods), 5), dtype=np.int8)
    cached = {}
    for m, method in enumerate(methods):
        if method == 'EXNEX':
            out[m] = engine.posterior(response_counts, n, p0, legacy_p0, exnex=True)
            labels[m] = -1
            continue
        z = canonical_labels(choices[method]); labels[m] = z
        if method == 'Independent':
            out[m] = independent_table(n, p0, legacy_p0)[response_counts]
            continue
        key = tuple(z)
        if key not in cached:
            answer = np.empty((5, len(FIELDS)))
            for g in np.unique(z):
                idx = np.flatnonzero(z == g)
                answer[idx] = engine.posterior(response_counts[idx], n, p0, legacy_p0)
            cached[key] = answer
        out[m] = cached[key]
    if not np.isfinite(out[..., :6]).all():
        raise ArithmeticError('Nonfinite Bayesian output.')
    return out, labels
