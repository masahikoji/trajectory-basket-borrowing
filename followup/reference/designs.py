"""Explicit source-controlled main, supplemental, and sensitivity DGMs."""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from scipy.optimize import brentq
from reference.core import BASE_P, BASE_PI, GROUPS, exact_truth, late_matrix, early_matrix, OBS_PROB

SUP_P=np.array([
 [[.60,0,0,.40],[.10,.30,.20,.40],[.03,.15,.47,.35],[0,0,.40,.60]],
 [[.45,0,0,.55],[.05,.30,.10,.55],[.01,.08,.41,.50],[0,.02,.08,.90]],
 BASE_P[1].copy()])
SUP_PI=np.array([[0,.05,.45,.50],[0,.04,.41,.55],[.05,.10,.35,.50]])
LEGACY_THRESHOLDS={'main':.467,'supplement':.2919}

def get_group_parameters(family,scenario,supp_source='code'):
    if family=='main': return BASE_PI.copy(),BASE_P.copy()
    if family!='supplement': raise ValueError(f'Unknown family {family}')
    pi,P=SUP_PI.copy(),SUP_P.copy()
    if supp_source=='manuscript' and scenario==3:
        P[0]=BASE_P[0]; P[2]=BASE_P[0]
    elif supp_source not in ('code','manuscript'): raise ValueError('Unknown supplementary source.')
    return pi,P

def reference_orr(family):
    pi,P=(BASE_PI[0],BASE_P[0]) if family=='main' else (SUP_PI[0],SUP_P[0])
    return float(exact_truth(pi,np.repeat(P[None],9,axis=0))['orr'])

@dataclass
class Experiment:
    name: str
    family: str
    scenario: int
    n: int
    dynamics: str
    supp_source: str
    initial: np.ndarray
    sequences: np.ndarray
    truth_labels: np.ndarray
    p0: float
    legacy_p0: float
    early_factors: np.ndarray
    late_odds: float
    early_transitions: int
    orr_matched: bool
    @property
    def truth(self):
        return [exact_truth(pi,seq) for pi,seq in zip(self.initial,self.sequences)]
    @property
    def seed_family(self): return 0 if self.family=='main' else (1 if self.supp_source=='code' else 2)

def make_experiment(family,scenario,n,supp_source='code',late_odds=1.,early_transitions=2,match_orr=True,threshold_mode='exact-reference'):
    pi,P=get_group_parameters(family,scenario,supp_source)
    seqs=[];factors=[]
    for j in range(3):
        target=float(exact_truth(pi[j],np.repeat(P[j][None],9,axis=0))['orr'])
        def sequence(lf):
            early=early_matrix(P[j],math.exp(lf));late=late_matrix(P[j],late_odds)
            return np.concatenate([np.repeat(early[None],early_transitions,axis=0),np.repeat(late[None],9-early_transitions,axis=0)])
        lf=0.
        if late_odds!=1 and match_orr:
            def f(x): return float(exact_truth(pi[j],sequence(x))['orr'])-target
            if f(-12)*f(12)>0: raise ValueError('Cannot match baseline ORR under requested dynamics.')
            lf=brentq(f,-12,12,xtol=1e-13)
        seqs.append(sequence(lf));factors.append(math.exp(lf))
    ids=GROUPS[scenario]
    dynamics='homogeneous' if late_odds==1 else f'late_PD_OR_{late_odds:g}'
    source=supp_source if family=='supplement' else 'main'
    name=f'{family}_{source}_S{scenario}_n{n}_{dynamics}'
    p0=reference_orr(family) if threshold_mode=='exact-reference' else LEGACY_THRESHOLDS[family]
    return Experiment(name,family,scenario,n,dynamics,source,pi[ids],np.array(seqs)[ids],ids.copy(),p0,
                      LEGACY_THRESHOLDS[family],np.array(factors)[ids],late_odds,early_transitions,match_orr)

def plan(suite='all',sizes=None,scenarios=(1,2,3),supp_source='code',late_odds=(1.5,2.),early_transitions=2,match_orr=True,threshold_mode='exact-reference'):
    result=[]
    if suite in ('all','main'):
        for n in (sizes or (20,30,50)):
            for s in scenarios: result.append(make_experiment('main',s,n,threshold_mode=threshold_mode))
    if suite in ('all','supplement'):
        for n in (sizes or (30,)):
            for s in scenarios: result.append(make_experiment('supplement',s,n,supp_source=supp_source,threshold_mode=threshold_mode))
    if suite in ('all','sensitivity'):
        for n in (sizes or (20,30,50)):
            for s in (1,2,3):
                if s not in scenarios: continue
                for factor in (1.,*late_odds):
                    result.append(make_experiment('main',s,n,late_odds=factor,early_transitions=early_transitions,
                                                  match_orr=match_orr,threshold_mode=threshold_mode))
    return list({e.name:e for e in result}.values())
