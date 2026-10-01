"""Cluster-count-preserving membership refinement for a selected two-group partition."""
from __future__ import annotations
import numpy as np
from scipy.special import i0e
from cluster_selection import (PatientData, ORDINAL, fitted_and_jackknife,
    information_partition, partition_geometry, _select_tie)

METHODS=('Baseline','Raw_mix50','Hitting_uniform','Hitting_mix25',
         'Hitting_mix50','Hitting_mix75')
PRIMARY='Hitting_mix50'

def fitted_hitting_probability(initial,transitions,lengths):
    """Probability of ever CR/PR, fitted to ALL observed transition counts.

    phi = 1 - sum_t w_t pi_N Q**(t-1) 1, N={SD,PD}.
    Arbitrary leading dimensions allow vectorized leave-one-patient-out fits.
    Empty outgoing rows are .25 each, exactly as in the frozen baseline.
    """
    ns=initial.sum(-1)
    if np.any(ns<=0):raise ValueError('Cannot fit empty baskets.')
    pi=initial/ns[...,None]; w=lengths/ns[...,None]; rows=transitions.sum(-1)
    P=np.divide(transitions,rows[...,None],out=np.full(transitions.shape,.25,dtype=float),where=rows[...,None]>0)
    non=pi[...,2:].copy(); Q=P[...,2:,2:]; survived=w[...,0]*non.sum(-1)
    for t in range(1,w.shape[-1]):
        non=np.einsum('...r,...rs->...s',non,Q)
        survived+=w[...,t]*non.sum(-1)
    return np.clip(1-survived,0,1)


def score_parts(means,pseudovalues,n,weight=0.):
    """Twice-log working directional evidence minus original size penalty.

    weight=0 reproduces the frozen baseline's orientation-average score.
    weight>0 uses an equal-family (or sensitivity-weighted) mixture of:
    (1) the original uniform direction distribution in whitened feature space;
    (2) the direction corresponding to the marginal ordinal-occupancy contrast.
    The spike direction a=S**(1/2)e1/sqrt(S[0,0]) has unit length, so
    (a'B a)= n sum_c |c| (sbar_c-sbar)^2/S[0,0].
    """
    if not 0<=weight<1:raise ValueError('Direction weight must be in [0,1).')
    means=np.asarray(means,float);ps=np.asarray(pseudovalues,float)
    if means.shape!=(5,2) or ps.shape!=(5,n,2):raise ValueError('Expected 5 baskets and 2 features.')
    cen=ps-ps.mean(1,keepdims=True)
    S=np.einsum('jid,jie->de',cen,cen)/(5*(n-1))
    ev,U=np.linalg.eigh((S+S.T)/2);scale=max(float(ev.max()),1e-15)
    keep=ev>scale*1e-10
    if not np.any(keep):
        if np.allclose(means,means[0],atol=1e-12,rtol=0):return -partition_geometry(n)[3]
        keep=np.ones(2,dtype=bool);ev=np.maximum(ev,1e-12)
    whitener=U[:,keep]/np.sqrt(np.maximum(ev[keep],1e-15))
    q=(means-means.mean(0))@whitener;parts,ks,H,pen=partition_geometry(n)
    B=n*np.einsum('ji,pjk,kl->pil',q,H,q)
    eig=np.maximum(np.linalg.eigvalsh((B+B.transpose(0,2,1))/2),0)
    if eig.shape[-1]==1:G=eig[:,0]
    else:
        x=(eig[:,1]-eig[:,0])/4
        G=.5*eig.sum(1)+2*(np.log(i0e(x))+x)
    if weight==0:return G-pen
    # If a marginal feature is constant, the ordinal component is noninformative.
    s=means[:,0]-means[:,0].mean()
    ordinal_variance=float(np.sum(U[0,keep]**2*np.maximum(ev[keep],1e-15)))
    A=n*np.einsum('j,pjk,k->p',s,H,s)/max(ordinal_variance,1e-15)
    mix=2*np.logaddexp(np.log(weight)+A/2,np.log1p(-weight)+G/2)
    return mix-pen


def keep_k_refine(base_labels,means,pseudovalues,n,weight,rng):
    """Refine only observed K=2; preserve K for every possible data set."""
    base_labels=np.asarray(base_labels,int)
    if len(np.unique(base_labels))!=2:return base_labels.copy()
    scores=score_parts(means,pseudovalues,n,weight)
    parts,ks,_,_=partition_geometry(n)
    scores=np.where(ks==2,scores,-np.inf)
    # On a numerical tie retain baseline membership whenever it is a maximizer.
    ii,jj=np.triu_indices(5,1)
    same=(parts[:,ii]==parts[:,jj])==((base_labels[ii]==base_labels[jj])[None])
    bi=np.flatnonzero(same.all(1))
    if len(bi)==1 and scores[bi[0]]>=scores.max()-1e-10:return base_labels.copy()
    return _select_tie(scores,parts,ks,rng)


def analyze(data:PatientData,rng:np.random.Generator):
    """All comparison methods analyze exactly the same observed trial."""
    data.validate();n=data.n
    _,occ,_,po=fitted_and_jackknife(data)
    s=occ@ORDINAL;pseudo_s=po@ORDINAL
    observed_orr=data.response.mean(1)
    base_means=np.column_stack((s,observed_orr))
    base_pseudo=np.stack((pseudo_s,data.response),axis=-1)
    base=information_partition(base_means,base_pseudo,n,rng)
    labels=[base]
    if len(np.unique(base))!=2:
        return np.array([base.copy() for _ in METHODS],dtype=np.int8), {'selected_k':len(np.unique(base))}
    I=data.initial.sum(1);T=data.transitions.sum(1);L=data.lengths.sum(1)
    hit=fitted_hitting_probability(I,T,L)
    lh=fitted_hitting_probability(I[:,None]-data.initial,T[:,None]-data.transitions,L[:,None]-data.lengths)
    ph=n*hit[:,None]-(n-1)*lh
    mh=np.column_stack((s,hit));psh=np.stack((pseudo_s,ph),axis=-1)
    labels.append(keep_k_refine(base,base_means,base_pseudo,n,.5,rng))
    for w in (0.,.25,.5,.75):labels.append(keep_k_refine(base,mh,psh,n,w,rng))
    labels=np.array(labels,dtype=np.int8)
    if any(len(np.unique(z))!=2 for z in labels):raise AssertionError('K preservation violated.')
    return labels, {'selected_k':2}
