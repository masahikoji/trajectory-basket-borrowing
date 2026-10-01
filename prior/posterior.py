"""Numerical quadrature for the logistic-normal hierarchy, conditional on a selected partition."""
from __future__ import annotations
import hashlib, json, math, os
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path
import numpy as np
import scipy
from scipy.integrate import simpson, cumulative_simpson
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
from scipy.special import expit, logit, gammaln, logsumexp
from scipy.stats import gamma, beta as beta_dist

VERSION = '2.0.0'
FIELDS = ('post_mean','lower90','upper90','prob_gt_null','prob_gt_legacy','post_sd','post_EX_probability')

@dataclass(frozen=True)
class Prior:
    mu0: float = 0.0
    sigma0_sq: float = 1.0
    alpha: float = 2.0
    beta: float = 1.0
    ex_probability: float = 0.5
    nex_mu: float = 0.0
    nex_sd: float = 1.0
    def validate(self):
        if not np.isfinite(list(asdict(self).values())).all():
            raise ValueError('Prior contains a nonfinite value.')
        if min(self.sigma0_sq,self.alpha,self.beta,self.nex_sd)<=0:
            raise ValueError('Prior variance, shape, rate, and NEX SD must be positive.')
        if not 0<=self.ex_probability<=1:
            raise ValueError('EX probability must lie in [0,1].')

def integral_weights(x):
    return simpson(np.eye(len(x)),x=x,axis=-1)

def atomic_json(path,value):
    path=Path(path); tmp=path.with_name(path.name+f'.tmp.{os.getpid()}')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False));os.replace(tmp,path)

def grid_key(prior,level):
    return hashlib.sha256(json.dumps({'prior':asdict(prior),'level':level,'version':VERSION,
        'engine_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'numpy':np.__version__,'scipy':scipy.__version__},sort_keys=True).encode()).hexdigest()[:20]

def build_grid(cache_dir, prior=Prior(), level=2, ns=(20,30,50)):
    """Parent-only initialization. Do not run two jobs writing the same cache root."""
    prior.validate()
    if level not in (1,2,3,4): raise ValueError('Quadrature level must be 1,2,3,4.')
    directory=Path(cache_dir)/grid_key(prior,level);directory.mkdir(parents=True,exist_ok=True)
    meta_path=directory/'grid.json'
    if not meta_path.exists():
        z=np.linspace(-8,8,80*level+1)
        wz=integral_weights(z)*np.exp(-z*z/2)/math.sqrt(2*math.pi)
        upper=max(4.,float(np.log(gamma.isf(1e-16,prior.alpha))))
        n_eta=int(np.ceil((upper+16)/(.2/level)))
        n_eta+=n_eta%2
        eta=np.linspace(-16,upper,n_eta+1)
        we=integral_weights(eta)*np.exp(prior.alpha*eta-np.exp(eta)-gammaln(prior.alpha))
        mu,tau=np.meshgrid(prior.mu0+math.sqrt(prior.sigma0_sq)*z,np.exp(eta)/prior.beta,indexing='ij')
        mu=mu.ravel();tau=tau.ravel();w=(wz[:,None]*we).ravel()
        scale=max(1.,math.sqrt(prior.sigma0_sq),math.sqrt(prior.beta/prior.alpha),prior.nex_sd)
        theta=prior.mu0+scale*np.sinh(np.linspace(-7,7,560*level+1))
        
        wt=integral_weights(theta)
        np.save(directory/'theta.npy',theta);np.save(directory/'weights.npy',wt)
        np.save(directory/'prior_weights.npy',w)
        tmp=directory/'kernel.tmp.npy'
        K=np.lib.format.open_memmap(tmp,mode='w+',dtype='float64',shape=(len(w),len(theta)))
        for begin in range(0,len(w),256):
            end=min(begin+256,len(w))
            K[begin:end]=np.exp(.5*np.log(tau[begin:end,None]/(2*math.pi))-.5*tau[begin:end,None]*(theta-mu[begin:end,None])**2)
        K.flush();del K;os.replace(tmp,directory/'kernel.npy')
        atomic_json(meta_path,{'version':VERSION,'prior':asdict(prior),'level':level,
                              'theta_min':float(theta[0]),'theta_max':float(theta[-1]),
                              'n_theta':len(theta),'n_hyper':len(w),'eta_min':-16,'eta_max':upper,
                              'prior_weight_sum':float(w.sum()),
                              'note':'Finite-grid numerical integration; validate resolution, not an exact posterior oracle.'})
    for n in sorted(set(ns)):
        if n<0: raise ValueError('n cannot be negative.')
        target=directory/f'table_n{n}.npz'
        if target.exists(): continue
        theta=np.load(directory/'theta.npy');wt=np.load(directory/'weights.npy')
        K=np.load(directory/'kernel.npy',mmap_mode='r')
        ll=np.arange(n+1)[:,None]*theta-n*np.logaddexp(0,theta)
        ll-=ll.max(axis=1,keepdims=True)
        like=np.exp(ll)
        I=(K@(like*wt).T).T
        knex=np.exp(-.5*((theta-prior.nex_mu)/prior.nex_sd)**2)/(math.sqrt(2*math.pi)*prior.nex_sd)
        Inex=(like*wt)@knex
        if not np.isfinite(I).all() or np.any(I<=0) or np.any(Inex<=0):
            raise ArithmeticError('Nonpositive/nonfinite integrated likelihood. Refine quadrature.')
        tmp=target.with_name(target.name+f'.tmp.{os.getpid()}')
        with tmp.open('wb') as f: np.savez(f,like=like,I=I,knex=knex,Inex=Inex)
        os.replace(tmp,target)
    return directory

class PosteriorEngine:
    def __init__(self,directory):
        self.directory=Path(directory)
        self.metadata=json.loads((self.directory/'grid.json').read_text())
        self.prior=Prior(**self.metadata['prior'])
        self.x=np.load(self.directory/'theta.npy');self.p=expit(self.x)
        self.wt=np.load(self.directory/'weights.npy')
        self.hw=np.load(self.directory/'prior_weights.npy')
        self.K=np.load(self.directory/'kernel.npy',mmap_mode='r')
        self.tables={}
    def table(self,n):
        if n not in self.tables:
            f=self.directory/f'table_n{n}.npz'
            if not f.exists(): raise FileNotFoundError(f'Missing integration table for n={n}; run parent grid preparation.')
            with np.load(f) as x: self.tables[n]={k:x[k].copy() for k in x.files}
        return self.tables[n]
    def posterior(self,counts,n,p0,legacy,exnex=False):
        counts=np.asarray(counts,dtype=int)
        if np.any(counts<0) or np.any(counts>n) or not len(counts): raise ValueError('Invalid response counts.')
        order=np.argsort(counts,kind='stable');inverse=np.argsort(order)
        ans=self._fit(tuple(int(x) for x in counts[order]),int(n),float(p0),float(legacy),bool(exnex))
        return ans[inverse].copy()
    @lru_cache(maxsize=8192)
    def _fit(self,counts,n,p0,legacy,exnex):
        tb=self.table(n);rr=np.array(counts,dtype=int)
        exw=self.prior.ex_probability if exnex else 1.0
        LI=exw*tb['I'][rr]+(1-exw)*tb['Inex'][rr,None]
        logLI=np.log(LI)
        logh=np.log(self.hw)+logLI.sum(axis=0)
        hn=np.exp(logh-logsumexp(logh))
        ans=[];unique={}
        for k,r in enumerate(rr):
            if int(r) in unique:
                ans.append(unique[int(r)]);continue
            lo=logh-logLI[k];v=np.exp(lo-lo.max())
            dens=exw*(v@self.K)+(1-exw)*v.sum()*tb['knex']
            pdf=dens*tb['like'][r]
            z=float(pdf@self.wt)
            if z<=0 or not np.isfinite(z): raise ArithmeticError('Invalid posterior normalizer.')
            pdf/=z
            cdf=np.r_[0.,cumulative_simpson(pdf,x=self.x)]
            if np.min(np.diff(cdf)) < -1e-9:
                raise ArithmeticError('Nonmonotone numerical CDF; use a finer --quad-level.')
            cdf=np.clip(np.maximum.accumulate(cdf),0,1);cdf/=cdf[-1]
            # Harmless floating underflow in remote, effectively flat tails is local.
            with np.errstate(over='ignore',divide='ignore',invalid='ignore'):
                curve=PchipInterpolator(self.x,cdf,extrapolate=False)
            q=[]
            for a in (.05,.95):
                hi=int(np.searchsorted(cdf,a,side='left'));hi=min(max(1,hi),len(cdf)-1)
                xx=brentq(lambda t:float(curve(t))-a,self.x[hi-1],self.x[hi],xtol=1e-11)
                q.append(float(expit(xx)))
            mean=float((pdf*self.p)@self.wt);second=float((pdf*self.p**2)@self.wt)
            tails=[float(np.clip(1-float(curve(logit(t))),0,1)) for t in (p0,legacy)]
            postex=float(hn@(exw*tb['I'][r]/LI[k])) if exnex else float('nan')
            row=np.array([mean,q[0],q[1],*tails,math.sqrt(max(0,second-mean*mean)),postex])
            if not np.isfinite(row[:6]).all(): raise ArithmeticError('Nonfinite posterior summary.')
            unique[int(r)]=row;ans.append(row)
        out=np.array(ans);out.setflags(write=False);return out

@lru_cache(maxsize=64)
def independent_table(n,p0,legacy):
    a=np.arange(n+1)+1.;b=n+2-a
    return np.column_stack((a/(a+b),beta_dist.ppf(.05,a,b),beta_dist.ppf(.95,a,b),
                            beta_dist.sf(p0,a,b),beta_dist.sf(legacy,a,b),
                            np.sqrt(a*b/((a+b)**2*(a+b+1))),np.full(n+1,np.nan)))
