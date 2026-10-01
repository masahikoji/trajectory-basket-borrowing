"""Observation-count distribution sensitivity designs for the main simulation.

Only the distribution of the number of response assessments is changed.
Transition mechanisms A/B/C, initial distributions, the primary analysis rule,
and the stage-2 prior are unchanged from suite 3.0.2.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import hashlib
import numpy as np
from reference.core import BASE_P, BASE_PI, GROUPS, exact_truth

PRIMARY_OBS_PROB = np.array([.03,.05,.25,.25,.20,.10,.05,.03,.02,.02], dtype=float)
SHORT_OBS_PROB = np.array([.03,.25,.25,.25,.10,.10,.02,0.,0.,0.], dtype=float)
LONG_OBS_PROB = np.array([.03,.05,.10,.20,.25,.20,.10,.03,.02,.02], dtype=float)
FOLLOWUP_PROFILES = {
    'primary': PRIMARY_OBS_PROB,
    'short': SHORT_OBS_PROB,
    'long': LONG_OBS_PROB,
}
EXPECTED_MEANS = {'primary':4.45, 'short':3.52, 'long':5.00}


def _validate_weights(name, w):
    w=np.asarray(w,float)
    if w.shape!=(10,) or not np.isfinite(w).all() or np.any(w<0) or not np.isclose(w.sum(),1.,atol=1e-12,rtol=0):
        raise ValueError(f'Invalid observation-count distribution for {name}.')
    mean=float(w@np.arange(1,11))
    if not np.isclose(mean, EXPECTED_MEANS[name], atol=1e-12, rtol=0):
        raise ValueError(f'Unexpected mean observation count for {name}: {mean}')
    return w

for _name,_w in FOLLOWUP_PROFILES.items(): _validate_weights(_name,_w)


def profile_null(profile):
    w=FOLLOWUP_PROFILES[profile]
    seq=np.repeat(BASE_P[0][None],9,axis=0)
    return float(exact_truth(BASE_PI[0],seq,weights=w)['orr'])


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
    source: str = 'main_followup_sensitivity'
    early_factors: np.ndarray | None = None
    late_odds: float = 1.0
    early_transitions: int = 0
    followup_profile: str = 'primary'
    observation_probabilities: np.ndarray | None = None
    stream_key: str = ''

    def validate(self):
        if self.n < 3 or self.initial.shape != (5,4) or self.sequences.shape != (5,9,4,4):
            raise ValueError('Requires five baskets, n>=3, ten possible observations, four states.')
        for array in (self.initial,self.sequences):
            if not np.isfinite(array).all() or np.any(array<0) or not np.allclose(array.sum(-1),1,rtol=0,atol=1e-12):
                raise ValueError('Invalid probability array in study design.')
        if self.labels.shape!=(5,) or not 0<self.p0<1 or not 0<self.legacy_p0<1:
            raise ValueError('Invalid truth labels or decision threshold.')
        if self.followup_profile not in FOLLOWUP_PROFILES:
            raise ValueError('Unknown follow-up profile.')
        if self.observation_probabilities is None:
            raise ValueError('Missing observation-count probabilities.')
        _validate_weights(self.followup_profile,self.observation_probabilities)
        if not self.stream_key:
            raise ValueError('Missing paired RNG stream key.')
        # A must be the exact null boundary; B/C must remain active.
        tp=self.true_orr
        if not np.isclose(tp[0],self.p0,atol=1e-12,rtol=0):
            raise ValueError('Mechanism A is not at the follow-up-specific null boundary.')
        active=self.labels>0
        if np.any(active) and not np.all(tp[active] > self.p0):
            raise ValueError('Mechanisms B/C must remain active relative to mechanism A.')

    @property
    def mean_assessments(self):
        return float(np.asarray(self.observation_probabilities)@np.arange(1,11))

    @property
    def true_orr(self):
        return np.array([exact_truth(pi,P,weights=self.observation_probabilities)['orr']
                         for pi,P in zip(self.initial,self.sequences)])

    def describe(self):
        out=asdict(self)
        out['mean_assessments']=self.mean_assessments
        return out


def stable_id(name):
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], 'little')


def settings(profiles=('short','long'), sizes=(20,30,50), scenarios=(1,2,3)):
    profiles=tuple(profiles);sizes=tuple(sizes);scenarios=tuple(scenarios)
    if not profiles or len(set(profiles))!=len(profiles) or any(x not in FOLLOWUP_PROFILES for x in profiles):
        raise ValueError('Profiles must be distinct values among primary, short, long.')
    if not sizes or len(set(sizes))!=len(sizes) or any(int(n)<3 for n in sizes):
        raise ValueError('Sample sizes must be distinct integers >=3.')
    if not scenarios or len(set(scenarios))!=len(scenarios) or any(int(s) not in (1,2,3) for s in scenarios):
        raise ValueError('Scenarios must be distinct values among 1,2,3.')
    seq_by_mech=np.repeat(BASE_P[:,None,:,:],9,axis=1)
    result=[]
    for profile in profiles:
        w=FOLLOWUP_PROFILES[profile].copy(); p0=profile_null(profile)
        for n in sizes:
            for s in scenarios:
                ids=GROUPS[int(s)]
                name=f'followup_{profile}_S{s}_n{n}'
                stream_key=f'main_main_S{s}_n{n}_homogeneous'
                result.append(Setting(
                    name=name,family='followup_sensitivity',scenario=str(s),n=int(n),
                    initial=BASE_PI[ids].copy(),sequences=seq_by_mech[ids].copy(),labels=ids.copy(),
                    description=(f'Main homogeneous mechanism; only observation-count distribution is {profile}. '
                                 f'Mean assessments={float(w@np.arange(1,11)):.2f}. Null threshold is exact ORR of mechanism A.'),
                    p0=p0,legacy_p0=.467,dynamics=f'followup_{profile}',source='main_followup_sensitivity',
                    early_factors=np.ones(5),late_odds=1.0,early_transitions=0,
                    followup_profile=profile,observation_probabilities=w,stream_key=stream_key))
    for e in result:e.validate()
    return result
