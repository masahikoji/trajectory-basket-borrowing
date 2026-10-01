"""Observed-data simulation and clustering. No true parameter enters estimated_features.
Inherited algorithm: average-linkage L1; 60 threshold candidates; singleton
silhouettes zero; fallback to one cluster when maximum silhouette < 0.25.
This explicitly follows the supplied executable rule, NOT the inconsistent
minimum-cluster-size wording in the manuscript.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist, squareform
CR, PR, SD, PD = range(4)
STATES = ("CR", "PR", "SD", "PD")
OBS_PROB = np.array([.03, .05, .25, .25, .20, .10, .05, .03, .02, .02])
OBS_POINTS = np.arange(1, 11)
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


def validate_probabilities(pi: np.ndarray, seq: np.ndarray,
                           weights: np.ndarray = OBS_PROB) -> None:
    if pi.shape != (4,) or seq.shape != (len(weights) - 1, 4, 4):
        raise ValueError("Expected pi shape (4,) and seq shape (Tmax-1,4,4).")
    for name, x in (("pi", pi), ("transition sequence", seq), ("weights", weights)):
        if not np.isfinite(x).all() or np.any(x < 0):
            raise ValueError(f"{name}: negative/nonfinite probabilities.")
    if not np.isclose(pi.sum(), 1, atol=1e-12, rtol=0):
        raise ValueError("Initial probabilities do not sum to one.")
    if not np.allclose(seq.sum(axis=-1), 1, atol=1e-12, rtol=0):
        raise ValueError("Transition rows do not sum to one.")
    if not np.isclose(weights.sum(), 1, atol=1e-12, rtol=0):
        raise ValueError("Assessment-count probabilities do not sum to one.")


def exact_truth(pi: np.ndarray, seq: np.ndarray,
                weights: np.ndarray = OBS_PROB) -> dict[str, np.ndarray | float]:
    """Exact finite-horizon probabilities, no MC and no model approximation.

    With Q_l = P_l[{SD,PD},{SD,PD}],
    ORR = 1 - sum_t w_t pi[{SD,PD}] Q_1 ... Q_(t-1) 1.
    Rows of pseudo_P are the risk-set/state-occupancy-weighted pooled limits.
    Independence of T and the potential response trajectory is part of this DGM.
    """
    pi, seq, weights = np.asarray(pi), np.asarray(seq), np.asarray(weights)
    validate_probabilities(pi, seq, weights)
    marginals = [pi.copy()]
    never_response = pi[2:].copy()
    hit_cdf = [1.0 - never_response.sum()]
    expected_counts = np.zeros((4, 4))
    for l, P in enumerate(seq):
        # l=0 is assessment 1 -> 2 and is observed only when T >= 2.
        at_risk_prob = weights[l + 1:].sum()
        expected_counts += at_risk_prob * marginals[-1][:, None] * P
        marginals.append(marginals[-1] @ P)
        never_response = never_response @ P[2:, 2:]
        hit_cdf.append(1.0 - never_response.sum())
    marginals = np.asarray(marginals)
    true_final = weights @ marginals
    row_totals = expected_counts.sum(axis=1)
    pooled = np.divide(expected_counts, row_totals[:, None],
                       out=np.full((4, 4), .25), where=row_totals[:, None] > 0)
    working_final = weighted_final(pi, pooled, weights)
    return {"orr": float(weights @ np.asarray(hit_cdf)),
            "final": true_final, "marginals": marginals,
            "hit_cdf": np.asarray(hit_cdf), "pooled_limit": pooled,
            "working_final_limit": working_final,
            "expected_outgoing_per_patient": row_totals}


def weighted_final(pi: np.ndarray, P: np.ndarray, weights: np.ndarray) -> np.ndarray:
    dist = np.array(pi, dtype=float, copy=True)
    out = weights[0] * dist
    for w in weights[1:]:
        dist = dist @ P
        out += w * dist
    return out


def late_matrix(P: np.ndarray, odds_multiplier: float) -> np.ndarray:
    """Increase PD vs non-PD odds for origins CR/PR/SD; leave PD row unchanged."""
    out = P.copy()
    out[:3, PD] *= odds_multiplier
    out[:3] /= out[:3].sum(axis=1, keepdims=True)
    return out


def early_matrix(P: np.ndarray, response_odds_multiplier: float) -> np.ndarray:
    """Tilt early SD -> {CR,PR} odds; preserve conditional CR:PR and SD:PD ratios."""
    out = P.copy()
    out[SD, :2] *= response_odds_multiplier
    out[SD] /= out[SD].sum()
    return out


@dataclass
class ObservedTrial:
    response_counts: np.ndarray
    initial_counts: np.ndarray
    transition_counts: np.ndarray
    final_counts: np.ndarray
    length_counts: np.ndarray
    n: int


def simulate_observed_trial(uniforms: np.ndarray, initial: np.ndarray,
                            sequences: np.ndarray,
                            weights: np.ndarray = OBS_PROB) -> ObservedTrial:
    """Generate ALL observed assessments, including those AFTER the first response.

    uniforms: (J,n,Tmax+1): count draw, initial draw, Tmax-1 transition draws.
    T is generated independently; this sensitivity does NOT test informative dropout.
    """
    J, n, width = uniforms.shape
    Tmax = len(weights)
    if width != Tmax + 1 or initial.shape != (J, 4) or sequences.shape != (J, Tmax - 1, 4, 4):
        raise ValueError("Invalid DGM array dimensions.")
    lengths = np.searchsorted(np.cumsum(weights), uniforms[:, :, 0], side="right") + 1
    states = (uniforms[:, :, 1, None] >= np.cumsum(initial, axis=-1)[:, None, :]).sum(axis=-1)
    states = np.minimum(states, 3).astype(np.int64)
    baskets = np.broadcast_to(np.arange(J)[:, None], (J, n))
    initial_counts = np.bincount((baskets * 4 + states).ravel(), minlength=J * 4).reshape(J, 4)
    length_counts = np.bincount((baskets * Tmax + lengths - 1).ravel(),
                               minlength=J * Tmax).reshape(J, Tmax)
    ever = states < 2
    counts = np.zeros((J, 4, 4), dtype=np.int64)
    for l in range(Tmax - 1):
        eligible = lengths >= l + 2
        bj, ij = np.nonzero(eligible)
        if not len(bj):
            break
        origins = states[bj, ij].copy()
        cdf = np.cumsum(sequences[bj, l, origins, :], axis=-1)
        destinations = (uniforms[bj, ij, l + 2, None] >= cdf).sum(axis=-1)
        destinations = np.minimum(destinations, 3)
        counts += np.bincount(bj * 16 + origins * 4 + destinations,
                              minlength=J * 16).reshape(J, 4, 4)
        states[bj, ij] = destinations
        ever[bj, ij] |= destinations < 2
    final_counts = np.bincount((baskets * 4 + states).ravel(), minlength=J * 4).reshape(J, 4)
    return ObservedTrial(ever.sum(axis=1).astype(int), initial_counts, counts,
                         final_counts, length_counts, n)


def estimated_features(data: ObservedTrial) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ANALYSIS only: no true pi, true P, group or time-varying matrices are inputs."""
    pi_hat = data.initial_counts / data.n
    rows = data.transition_counts.sum(axis=2)
    P_hat = np.divide(data.transition_counts, rows[:, :, None],
                      out=np.full(data.transition_counts.shape, .25, dtype=float),
                      where=rows[:, :, None] > 0)
    w_hat = data.length_counts / data.n
    dist = pi_hat.copy()
    finals = w_hat[:, 0, None] * dist
    for t in range(1, w_hat.shape[1]):
        dist = np.einsum("jr,jrs->js", dist, P_hat)
        finals += w_hat[:, t, None] * dist
    X = np.column_stack((finals, data.response_counts / data.n))
    return X, P_hat, rows


def canonical_labels(labels: np.ndarray) -> np.ndarray:
    mapping: dict[int, int] = {}
    return np.array([mapping.setdefault(int(x), len(mapping)) for x in labels], dtype=int)


def signature(labels: np.ndarray) -> str:
    z = canonical_labels(labels)
    return "|".join("".join(str(i + 1) for i in np.flatnonzero(z == g))
                    for g in range(z.max() + 1))


def adjusted_rand_value(truth: np.ndarray, labels: np.ndarray) -> float:
    _, a = np.unique(truth, return_inverse=True)
    _, b = np.unique(labels, return_inverse=True)
    cross = np.bincount(a * (b.max() + 1) + b, minlength=(a.max() + 1) * (b.max() + 1)).reshape(a.max() + 1, b.max() + 1)
    choose2 = lambda x: float(np.sum(x * (x - 1) / 2))
    total = len(labels) * (len(labels) - 1) / 2
    if total == 0:
        return 1.0
    common, row, col = choose2(cross), choose2(cross.sum(axis=1)), choose2(cross.sum(axis=0))
    expected = row * col / total
    denominator = .5 * (row + col) - expected
    return 1.0 if abs(denominator) < 1e-14 else (common - expected) / denominator


def silhouette_mean(D: np.ndarray, labels: np.ndarray) -> float:
    """Standard average silhouette with singleton samples assigned zero.

    Equivalent to sklearn silhouette_score(metric='precomputed'); implemented
    directly to avoid loading sklearn/pandas in every spawn worker.
    """
    groups = np.unique(labels)
    values = np.zeros(len(labels))
    for i, label in enumerate(labels):
        own = labels == label
        if own.sum() <= 1:
            continue
        a = float(D[i, own].sum() / (own.sum() - 1))
        b = min(float(D[i, labels == g].mean()) for g in groups if g != label)
        denominator = max(a, b)
        values[i] = (b - a) / denominator if denominator > 0 else 0.0
    return float(values.mean())


@lru_cache(maxsize=1024)
def partition_metrics(labels: tuple[int, ...], truth: tuple[int, ...]) -> tuple[str, int, float]:
    z = np.asarray(labels)
    return signature(z), len(np.unique(z)), float(adjusted_rand_value(np.asarray(truth), z))


def choose_partition(X: np.ndarray, guard: float = .25,
                     grid_size: int = 60) -> tuple[np.ndarray, float]:
    """Retain the source's 60-point threshold-grid / average-linkage silhouette rule.

    Compute a full tree ONCE rather than refitting it 60 times. Singleton samples
    contribute zero to the silhouette mean, as in sklearn and the supplied code.
    Singleton clusters are permitted.
    """
    J = len(X)
    d = pdist(X, metric="cityblock")
    if np.allclose(d, 0):
        return np.zeros(J, dtype=int), 0.0
    dmin, dmax = np.percentile(d, [5, 95])
    if dmax <= dmin:
        dmax, dmin = d.max(), d.min()
    thresholds = np.linspace(max(1e-6, dmin * .5), dmax * 1.5, grid_size)
    D = squareform(d)
    tree = linkage(d, method="average")
    children = tree[:, :2].astype(int)
    heights = tree[:, 2]
    members: list[list[int]] = [[i] for i in range(J)]
    for a, b in children:
        members.append(members[a] + members[b])
    best_score = -np.inf
    best_labels = np.zeros(J, dtype=int)
    cache: dict[tuple[int, ...], float] = {}
    for th in thresholds:
        labels = np.arange(J)
        for h, (a, b) in enumerate(children):
            if heights[h] >= th:
                break
            labels[members[J + h]] = J + h
        labels = canonical_labels(labels)
        k = len(np.unique(labels))
        if not 1 <= k <= J - 1:
            continue
        key = tuple(labels)
        if key not in cache:
            cache[key] = 0.0 if k == 1 else silhouette_mean(D, labels)
        score = cache[key]
        if score > best_score:
            best_score, best_labels = score, labels.copy()
    if not np.isfinite(best_score):
        raise ArithmeticError("No admissible partition on the distance threshold grid.")
    if best_score < guard:  # retain the source's strict inequality
        best_labels = np.zeros(J, dtype=int)
    return best_labels, float(best_score)


