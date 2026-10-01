"""Main-study response mechanisms and transition-count estimation."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import numpy as np

STATES = ('CR', 'PR', 'SD', 'PD')
OBS_PROB = np.array([.03, .05, .25, .25, .20, .10, .05, .03, .02, .02])
BASE_P = np.array([
    [[.60, .00, .00, .40], [.10, .40, .10, .40],
     [.05, .20, .40, .35], [.00, .05, .35, .60]],
    [[.675, .000, .000, .325], [.150, .475, .100, .275],
     [.075, .250, .400, .275], [.025, .100, .325, .550]],
    [[.75, .00, .00, .25], [.20, .55, .10, .15],
     [.10, .30, .40, .20], [.05, .15, .30, .50]],
])
BASE_PI = np.array([[.05, .10, .35, .50], [.075, .150, .425, .350],
                    [.10, .20, .50, .20]])
GROUPS = {1: np.array([0, 0, 0, 0, 0]),
          2: np.array([0, 0, 0, 1, 1]),
          3: np.array([0, 0, 1, 1, 2])}
FIELDS = ('transition_counts', 'initial_counts', 'final_counts',
          'length_counts', 'responses', 'Phat')
DEFAULT_SEED = 610202610


@dataclass
class Setting:
    name: str
    scenario: int
    n: int
    initial: np.ndarray
    sequences: np.ndarray
    mechanisms: np.ndarray

    @property
    def truth(self):
        return BASE_P[self.mechanisms].copy()

    def describe(self):
        return dict(name=self.name, scenario=self.scenario, n=self.n,
                    initial=self.initial.tolist(), sequences=self.sequences.tolist(),
                    mechanisms=self.mechanisms.tolist(),
                    observation_count_probabilities=OBS_PROB.tolist())


def main_settings(sizes=(20, 30, 50), scenarios=(1, 2, 3)):
    if not sizes or not scenarios or len(set(sizes)) != len(sizes) or len(set(scenarios)) != len(scenarios):
        raise ValueError('Provide nonempty, distinct sizes and scenarios.')
    if any(n not in (20, 30, 50) for n in sizes) or any(s not in GROUPS for s in scenarios):
        raise ValueError('This study supports Main S1-S3 at n=20,30,50 only.')
    sequences = []
    for P in BASE_P:
        early = P.copy(); early[2] /= early[2].sum()
        late = P.copy(); late[:3] /= late[:3].sum(axis=1, keepdims=True)
        sequences.append(np.concatenate([np.repeat(early[None], 2, axis=0),
                                         np.repeat(late[None], 7, axis=0)]))
    sequences = np.array(sequences)
    return [Setting(f'main_main_S{s}_n{n}_homogeneous', s, n,
                    BASE_PI[GROUPS[s]].copy(), sequences[GROUPS[s]].copy(), GROUPS[s].copy())
            for n in sizes for s in scenarios]


def stable_id(name):
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], 'little')


def estimate_transition(counts):
    counts = np.asarray(counts)
    if counts.shape[-2:] != (4, 4) or not np.isfinite(counts).all() or np.any(counts < 0):
        raise ValueError('Expected finite, nonnegative 4-by-4 transition counts.')
    rows = counts.sum(-1)
    return np.divide(counts, rows[..., None],
                     out=np.full(counts.shape, .25, dtype=float),
                     where=rows[..., None] > 0)


def generate_counts(uniforms, initial, sequences):
    J, n, width = uniforms.shape
    if J != 5 or width != 11 or initial.shape != (5, 4) or sequences.shape != (5, 9, 4, 4):
        raise ValueError('Invalid generating array dimensions.')
    lengths = np.searchsorted(np.cumsum(OBS_PROB), uniforms[:, :, 0], side='right') + 1
    states = (uniforms[:, :, 1, None] >= np.cumsum(initial, axis=-1)[:, None, :]).sum(-1)
    first = np.eye(4, dtype=np.int32)[states].sum(1)
    counts = np.zeros((J, 4, 4), dtype=np.int32)
    ever = states < 2
    for ell in range(9):
        bj, ij = np.nonzero(lengths >= ell + 2)
        origins = states[bj, ij].copy()
        cdf = np.cumsum(sequences[bj, ell, origins, :], axis=-1)
        dest = (uniforms[bj, ij, ell + 2, None] >= cdf).sum(-1)
        states[bj, ij] = dest
        np.add.at(counts, (bj, origins, dest), 1)
        ever[bj, ij] |= dest < 2
    return dict(transition_counts=counts, initial_counts=first,
                final_counts=np.eye(4, dtype=np.int32)[states].sum(1),
                length_counts=np.eye(10, dtype=np.int32)[lengths-1].sum(1),
                responses=ever.sum(1).astype(np.int32), Phat=estimate_transition(counts))


def sample_trial(setting, replicate, seed=DEFAULT_SEED):
    rng = np.random.default_rng(np.random.SeedSequence([seed, stable_id(setting.name), replicate, 0]))
    order = rng.permutation(5)
    result = generate_counts(rng.random((5, setting.n, 11)),
                             setting.initial[order], setting.sequences[order])
    inv = np.argsort(order)
    return {k: v[inv] for k, v in result.items()}


def validate_arrays(arrays, n, expected_reps=None):
    missing = set(FIELDS) - set(arrays)
    if missing:
        raise ValueError(f'Missing count fields: {sorted(missing)}')
    R = len(arrays['transition_counts'])
    if expected_reps is not None and R != expected_reps:
        raise ValueError('Unexpected number of saved trials.')
    shapes = dict(transition_counts=(R, 5, 4, 4), initial_counts=(R, 5, 4),
                  final_counts=(R, 5, 4), length_counts=(R, 5, 10),
                  responses=(R, 5), Phat=(R, 5, 4, 4))
    for k, shape in shapes.items():
        a = arrays[k]
        if a.shape != shape or not np.isfinite(a).all() or np.any(a < 0):
            raise ValueError(f'Invalid saved array: {k}')
        if k != 'Phat' and not np.array_equal(a, np.rint(a)):
            raise ValueError(f'Noninteger observed count: {k}')
    for k in ('initial_counts', 'final_counts', 'length_counts'):
        if not np.all(arrays[k].sum(-1) == n):
            raise ValueError(f'Patient totals do not equal n in {k}.')
    if np.any(arrays['responses'] > n):
        raise ValueError('Response count exceeds n.')
    trans = arrays['transition_counts']
    if not np.array_equal(trans.sum((-1, -2)), arrays['length_counts'] @ np.arange(10)):
        raise ValueError('Transition totals and assessment counts disagree.')
    if not np.array_equal(arrays['initial_counts'] + trans.sum(-2),
                           arrays['final_counts'] + trans.sum(-1)):
        raise ValueError('State flow conservation failed.')
    fitted = estimate_transition(trans)
    delta = float(np.max(np.abs(fitted - arrays['Phat'])))
    if delta > 5e-13:
        raise ValueError(f'Saved Phat is inconsistent with observed counts: maximum error {delta:g}.')
    return delta


def theoretical_support(pi, P, n):
    outgoing = np.zeros(4); visits = np.zeros(4)
    no_outgoing = np.zeros(4)
    for t, weight in enumerate(OBS_PROB, start=1):
        dist = pi.copy()
        for ell in range(t):
            visits += weight * dist
            if ell < t-1:
                outgoing += weight * dist
            dist = dist @ P
        for r in range(4):
            if t == 1:
                q = 1.
            else:
                ix = np.arange(4) != r
                q = float(pi[ix] @ np.linalg.matrix_power(P[np.ix_(ix, ix)], t-2) @ np.ones(3))
            no_outgoing[r] += weight * q
    return dict(expected_outgoing=n*outgoing, expected_visits=n*visits,
                empty_probability=no_outgoing**n)
