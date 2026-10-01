"""State-occupancy features, patient-level jackknife covariance, and partition selection."""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
from scipy.special import i0e
from reference.core import ObservedTrial, estimated_features, choose_partition

ORDINAL = np.array([1., 2./3., 1./3., 0.])
RESPONSE = np.array([1., 1., 0., 0.])
NON_PD = np.array([1., 1., 1., 0.])
METHODS = ('Corrected_final_silhouette', 'ORR_silhouette',
           'Final_ORR_IC', 'Occupancy_ORR_IC',
           'Occupancy_ORR_gate10', 'Occupancy_ORR_gate05',
           'Empirical_occupancy_ORR_IC', 'Response_occupancy_ORR_IC',
           'NonPD_occupancy_ORR_IC', 'Scalar_occupancy_IC')

@dataclass
class PatientData:
    initial: np.ndarray       # J x n x 4, one-hot first observed state
    transitions: np.ndarray   # J x n x 4 x 4, all observed transitions
    lengths: np.ndarray       # J x n x Tmax, one-hot number of observations
    occupancy: np.ndarray     # J x n x 4, each patient's proportions of observed states
    final: np.ndarray         # J x n x 4, one-hot last observed state
    response: np.ndarray      # J x n, indicator of ever CR or PR

    @property
    def n(self) -> int:
        return self.initial.shape[1]

    def aggregate(self) -> ObservedTrial:
        return ObservedTrial(self.response.sum(1).astype(int), self.initial.sum(1),
                             self.transitions.sum(1), self.final.sum(1),
                             self.lengths.sum(1), self.n)

    def validate(self) -> None:
        J,n,d=self.initial.shape
        if J!=5 or n<3 or d!=4:
            raise ValueError('This evaluator requires five baskets, n >= 3, four states.')
        if self.transitions.shape!=(J,n,4,4) or self.occupancy.shape!=(J,n,4) or self.final.shape!=(J,n,4) or self.response.shape!=(J,n):
            raise ValueError('Inconsistent patient summary dimensions.')
        for a in (self.initial,self.transitions,self.lengths,self.occupancy,self.final,self.response):
            if not np.isfinite(a).all() or np.any(a<0):
                raise ValueError('Negative or nonfinite observed summary.')
        for a in (self.initial,self.lengths,self.occupancy,self.final):
            if not np.allclose(a.sum(-1),1,rtol=0,atol=1e-12):
                raise ValueError('Observed state/length distributions must sum to one.')


def fitted_distributions(initial,transitions,lengths):
    """Pooled Markov MLEs; return final and mean-over-observed-visits distributions.

    Arbitrary leading dimensions are allowed, for vectorized leave-one-patient-out
    fits. An empty outgoing row retains the original uniform replacement.
    """
    ns=initial.sum(-1)
    if np.any(ns<=0): raise ValueError('Cannot fit empty baskets.')
    pi=initial/ns[...,None];weights=lengths/ns[...,None]
    rows=transitions.sum(-1)
    P=np.divide(transitions,rows[...,None],out=np.full(transitions.shape,.25,dtype=float),where=rows[...,None]>0)
    dist=pi.copy();cum=pi.copy()
    final=weights[...,0,None]*pi;occupancy=final.copy()
    for t in range(1,weights.shape[-1]):
        dist=np.einsum('...r,...rs->...s',dist,P)
        cum+=dist
        final+=weights[...,t,None]*dist
        occupancy+=weights[...,t,None]*cum/(t+1)
    return final,occupancy


def fitted_and_jackknife(data: PatientData):
    I=data.initial.sum(1);T=data.transitions.sum(1);L=data.lengths.sum(1)
    final,occ=fitted_distributions(I,T,L)
    lf,lo=fitted_distributions(I[:,None]-data.initial,T[:,None]-data.transitions,L[:,None]-data.lengths)
    n=data.n
    return final,occ,n*final[:,None]-(n-1)*lf,n*occ[:,None]-(n-1)*lo


@lru_cache(maxsize=8)
def partition_geometry(n: int):
    def build(z):
        if len(z)==5:
            yield tuple(z);return
        for a in range(max(z)+2): yield from build(z+[a])
    parts=np.array(list(build([0])),dtype=int)
    ks=parts.max(1)+1
    # Weighted between-cluster projection matrices after removing grand mean.
    H=np.empty((52,5,5));pen=np.empty(52)
    for a,z in enumerate(parts):
        H[a]=-.2
        pen[a]=-np.log(5*n)
        for g in range(ks[a]):
            ix=np.flatnonzero(z==g)
            H[a][np.ix_(ix,ix)]+=1/len(ix)
            pen[a]+=np.log(n*len(ix))
    for x in (parts,ks,H,pen):x.setflags(write=False)
    return parts,ks,H,pen


def _select_tie(scores,parts,ks,rng):
    ix=np.flatnonzero(scores >= scores.max()-1e-10)
    # Prefer smaller K on numerical ties; no preference for a known true partition.
    ix=ix[ks[ix]==ks[ix].min()]
    best=ix[0] if len(ix)==1 else rng.choice(ix)
    return parts[best].copy()


def information_partition(means,pseudovalues,n,rng):
    """Orientation-averaged, rank-one Gaussian working IC in <=2 dimensions.

    Pseudovalues are centered WITHIN each basket before covariance pooling.
    This covariance is an uncertainty estimate, not the covariance of basket means.
    Cluster means vary along an unknown line. Integrating line orientation (uniform
    on the whitened unit circle) avoids fixing a mixture of ORR and ordinal scores.
    Local-sample-size BIC-type penalty: sum_c log(n*|c|) - log(5*n).
    Not an exact Bayesian posterior, nor a frequentist error-control theorem.
    """
    J,d=means.shape
    if J!=5 or d not in (1,2):raise ValueError('Expected five baskets and one/two features.')
    cen=pseudovalues-pseudovalues.mean(1,keepdims=True)
    Sigma=np.einsum('jid,jie->de',cen,cen)/(J*(n-1))
    ev,U=np.linalg.eigh((Sigma+Sigma.T)/2)
    scale=max(float(ev.max()),1e-15)
    keep=ev>scale*1e-10
    if not np.any(keep):
        # Degenerate data: exact equal summaries merge, otherwise use a numerical
        # variance floor so constant but different observations can be separated.
        if np.allclose(means,means[0],atol=1e-12,rtol=0):return np.zeros(5,int)
        keep=np.ones(d,dtype=bool);ev=np.maximum(ev,1e-12)
    whitener=U[:,keep]/np.sqrt(np.maximum(ev[keep],1e-15))
    q=(means-means.mean(0))@whitener
    parts,ks,H,pen=partition_geometry(n)
    B=n*np.einsum('ji,pjk,kl->pil',q,H,q)
    eig=np.maximum(np.linalg.eigvalsh((B+B.transpose(0,2,1))/2),0.)
    if eig.shape[-1]==1:
        explained=eig[:,0]
    else:
        x=(eig[:,1]-eig[:,0])/4
        # 2 log E_direction exp(v' B v / 2), evaluated stably.
        explained=.5*eig.sum(1)+2*(np.log(i0e(x))+x)
    scores=explained-pen
    return _select_tie(scores,parts,ks,rng)


def permutation_homogeneity_gate(patient_features,rng,permutations=199):
    """Monte Carlo permutation p-value, jointly shuffling WHOLE patients.

    Conditional finite-sample validity requires exchangeability of the joint
    patient summaries across baskets (including their assessment mechanism).
    Same ORR alone does not imply this null. Gate controls only P(Khat>1) under
    that full-exchangeability null, not efficacy type I error or partial-null errors.
    """
    if permutations<19:raise ValueError('Use at least 19 permutations.')
    J,n,d=patient_features.shape
    v=patient_features.reshape(J*n,d).copy();v-=v.mean(0)
    total_cov=v.T@v/(J*n-1)
    inv=np.linalg.pinv(total_cov,hermitian=True,rcond=1e-10)
    m=v.reshape(J,n,d).mean(1)
    statistic=n*np.einsum('jd,de,je->',m,inv,m)
    if statistic<=1e-14:return 1.
    index=rng.permuted(np.broadcast_to(np.arange(J*n),(permutations,J*n)),axis=1)
    pm=v[index].reshape(permutations,J,n,d).mean(2)
    reference=n*np.einsum('bjd,de,bje->b',pm,inv,pm)
    ge=int(np.sum(reference>=statistic-1e-12))
    return (ge+1)/(permutations+1)


def analyze_trial(data: PatientData, rng: np.random.Generator, permutations: int=199):
    """The one entry point for inference. Only observed data and RNG enter here."""
    n=data.n;agg=data.aggregate()
    original,_,_=estimated_features(agg)
    z0,_=choose_partition(original);zr,_=choose_partition((agg.response_counts/n)[:,None])
    final,occ,pf,po=fitted_and_jackknife(data)
    r=data.response.mean(1);binary=data.response.astype(float)
    def joint(mu,pseudo):
        return information_partition(np.column_stack((mu,r)),np.stack((pseudo,binary),axis=-1),n,rng)
    zfinal=joint(final@ORDINAL,pf@ORDINAL)
    zocc=joint(occ@ORDINAL,po@ORDINAL)
    empirical=data.occupancy@ORDINAL
    zemp=joint(empirical.mean(1),empirical)
    zresponse=joint(occ@RESPONSE,po@RESPONSE)
    znonpd=joint(occ@NON_PD,po@NON_PD)
    zscalar=information_partition((occ@ORDINAL)[:,None],(po@ORDINAL)[:,:,None],n,rng)
    p=permutation_homogeneity_gate(np.stack((empirical,binary),axis=-1),rng,permutations)
    gate10=zocc.copy() if p<=.1 else np.zeros(5,int)
    gate05=zocc.copy() if p<=.05 else np.zeros(5,int)
    labels=np.array([z0,zr,zfinal,zocc,gate10,gate05,zemp,zresponse,znonpd,zscalar],dtype=np.int8)
    return labels,{'gate_p':p,'response_counts':agg.response_counts,
                   'markov_occ_score':occ@ORDINAL,
                   'empty_rows':int(np.sum(agg.transition_counts.sum(-1)==0))}
