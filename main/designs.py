"""Study plan. No simulation setting is supplied to an analysis function.

Primary plan: original main (9), supplemental code version (3), and time-nonhomogeneity sensitivity across S1-S3 and n=20,30,50 (18 nonhomogeneous settings; homogeneous baselines are the 9 main settings). The four previously evaluated stress settings are available explicitly.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import hashlib
import numpy as np
from reference.core import BASE_P, BASE_PI, OBS_PROB, exact_truth, validate_probabilities
from reference.designs import plan as original_plan, reference_orr, LEGACY_THRESHOLDS

@dataclass
class Setting:
    name: str
    family: str
    scenario: str
    n: int
    initial: np.ndarray
    sequences: np.ndarray
    labels: np.ndarray
    description: str
    p0: float
    legacy_p0: float
    dynamics: str = 'homogeneous'
    source: str = 'main'
    early_factors: np.ndarray | None = None
    late_odds: float = 1.0
    early_transitions: int = 2

    def validate(self):
        if self.n < 3 or self.initial.shape != (5, 4) or self.sequences.shape != (5, 9, 4, 4):
            raise ValueError('Requires five baskets, n>=3, ten possible observations, four states.')
        for array in (self.initial, self.sequences):
            if not np.isfinite(array).all() or np.any(array < 0) or not np.allclose(array.sum(-1), 1, rtol=0, atol=1e-12):
                raise ValueError('Invalid probability array in study design.')
        if self.labels.shape != (5,) or not 0 < self.p0 < 1 or not 0 < self.legacy_p0 < 1:
            raise ValueError('Invalid truth labels or decision threshold.')

    def describe(self):
        out = asdict(self)
        out['observation_count_probabilities'] = OBS_PROB
        return out

    @property
    def true_orr(self):
        return np.array([exact_truth(pi, P)['orr'] for pi, P in zip(self.initial, self.sequences)])


def stable_id(name):
    """Preserve the prior refinement evaluation's per-setting random streams."""
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], 'little')


def settings(suite='all', sizes=None, scenarios=(1, 2, 3), supp_source='code',
             include_stress=False, threshold_mode='exact-reference'):
    if suite not in ('all', 'main', 'supplement', 'sensitivity', 'stress'):
        raise ValueError('Unknown suite.')
    if sizes is not None and (len(set(sizes)) != len(sizes) or any(n < 3 for n in sizes)):
        raise ValueError('Sample sizes must be distinct integers >=3.')
    result = []
    if suite != 'stress':
        for e in original_plan(suite, sizes=sizes, scenarios=scenarios,
                               supp_source=supp_source, threshold_mode=threshold_mode):
            family = e.family if e.late_odds == 1 else 'sensitivity'
            result.append(Setting(e.name, family, str(e.scenario), e.n, e.initial,
                                  e.sequences, e.truth_labels,
                                  e.dynamics + '; supplemental source=' + supp_source,
                                  e.p0, e.legacy_p0, e.dynamics, e.supp_source,
                                  e.early_factors, e.late_odds, e.early_transitions))
    if suite == 'stress' or include_stress:
        p0 = reference_orr('main') if threshold_mode == 'exact-reference' else LEGACY_THRESHOLDS['main']
        kwargs = dict(p0=p0, legacy_p0=LEGACY_THRESHOLDS['main'], source='previous_stress')
        for k in (1, 2):
            name = 'stress_homogeneous_' + chr(65 + k)
            result.append(Setting(name, 'stress', 'homogeneous_' + chr(65+k), 30,
                    np.repeat(BASE_PI[k, None], 5, axis=0), np.tile(BASE_P[k], (5, 9, 1, 1)),
                    np.zeros(5, int), 'All five baskets share main ' + chr(65+k) +
                    ' dynamics. Trajectory homogeneity does NOT imply the efficacy null. Main reference p0 used.', **kwargs))
        strength = np.array([-5., -2., 0., 2., 5.])
        prob = np.exp(strength[:, None] * np.arange(3, -1, -1)[None]); prob /= prob.sum(1, keepdims=True)
        pp = np.repeat(prob[:, None, :], 4, axis=1)
        result.append(Setting('stress_five_groups', 'stress', 'five_groups', 30, prob,
                     np.repeat(pp[:, None], 9, axis=1), np.arange(5),
                     'Five separated iid-state processes. Main reference p0 used for descriptive efficacy outputs.', **kwargs))
        pi = np.array([.05, .15, .30, .50]); rho = np.array([.1, .1, .1, .85, .85])
        pp = rho[:, None, None] * np.eye(4) + (1-rho[:, None, None]) * np.tile(pi, (5, 4, 1))
        result.append(Setting('stress_equal_occupancy_persistence', 'stress', 'equal_occupancy_persistence', 30,
                      np.tile(pi, (5, 1)), np.repeat(pp[:, None], 9, axis=1), np.array([0,0,0,1,1]),
                      'Identical stationary distributions, persistence rho .1 versus .85. Main reference p0 used.', **kwargs))
    result = list({e.name: e for e in result}.values())
    for e in result: e.validate()
    return result
