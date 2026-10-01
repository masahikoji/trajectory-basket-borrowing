#!/usr/bin/env python3
"""Independent numerical checks; no simulation-performance claim.

Compares hierarchical/EXNEX posterior summaries at quadrature levels 2 and 3.
Also checks one-basket results against an independent collapsed-prior calculation:
integrating Gamma precision gives a Student-t(df=2*alpha) prior on theta-mu;
integrate mu by Gauss-Hermite, then integrate theta adaptively on the real line.
"""
from __future__ import annotations
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
import argparse,json,math,time
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.special import roots_hermitenorm,expit,logit
from scipy.stats import t as t_dist,norm
from scipy.optimize import brentq
from posterior import Prior,build_grid,PosteriorEngine
from reporting import write_csv

def singleton_reference(r,n,p0,prior=Prior(),exnex=False):
    x,w=roots_hermitenorm(160);mu=prior.mu0+math.sqrt(prior.sigma0_sq)*x;w/=math.sqrt(2*math.pi)
    scale=math.sqrt(prior.beta/prior.alpha)
    shift=(r*np.log(r/n)+(n-r)*np.log1p(-r/n)) if 0<r<n else 0.
    def density(th):
        pr=float(w@t_dist.pdf((th-mu)/scale,df=2*prior.alpha))/scale
        if exnex:pr=prior.ex_probability*pr+(1-prior.ex_probability)*norm.pdf(th,prior.nex_mu,prior.nex_sd)
        return pr*math.exp(r*th-n*np.logaddexp(0,th)-shift)
    def integrate_to(fun,b):
        # Split around the posterior mass. A single (-inf, 1000) quad call
        # can miss a narrow integrand near zero and incorrectly return zero.
        edges=[-np.inf]+[v for v in (-30.,-12.,-6.,0.,6.,12.,30.) if v<b]+[b]
        return sum(quad(fun,a,z,epsabs=1e-11,epsrel=5e-10,limit=250)[0]
                   for a,z in zip(edges[:-1],edges[1:]))
    def integral(b): return integrate_to(density,b)
    Z=integral(np.inf)
    mean=integrate_to(lambda th:expit(th)*density(th),np.inf)/Z
    qs=[expit(brentq(lambda th:integral(th)/Z-q,-100.,100.,xtol=1e-10)) for q in [.05,.95]]
    return np.array([mean,*qs,1-integral(logit(p0))/Z])

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',default='.posterior_cache');p.add_argument('--out',default='numerical_validation')
    p.add_argument('--sizes',nargs='+',type=int,default=[20,30,50]);p.add_argument('--random-cases',type=int,default=10)
    p.add_argument('--tolerance',type=float,default=2e-5)
    a=p.parse_args(argv);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    t0=time.monotonic();low=PosteriorEngine(build_grid(a.cache,Prior(),2,a.sizes));high=PosteriorEngine(build_grid(a.cache,Prior(),3,a.sizes))
    rng=np.random.default_rng(911);rows=[]
    for n in a.sizes:
        fixed=[[0],[n],[n//2],[0]*5,[n]*5,[n//2]*5,[0,1,n-1,n],[n//3,n//2,2*n//3],
               [int(.46*n)]*3+[int(.64*n)]*2,[int(.29*n),int(.29*n),int(.16*n),int(.16*n),int(.6*n)]]
        cases=fixed+[rng.binomial(n,rng.uniform(.05,.95),size=int(rng.integers(1,6))).tolist() for _ in range(a.random_cases)]
        for ex in [False,True]:
            for counts in cases:
                x=low.posterior(counts,n,.4672661234803711,.467,ex);y=high.posterior(counts,n,.4672661234803711,.467,ex)
                diff=float(np.max(np.abs(x[:,:6]-y[:,:6])))
                rows.append(dict(test='resolution_2_vs_3',n=n,method='EXNEX' if ex else 'HB',counts=str(counts),max_abs_difference=diff,passed=diff<a.tolerance))
            for r in [0,n//2,n]:
                x=low.posterior([r],n,.4672661234803711,.467,ex)[0,:4]
                y=singleton_reference(r,n,.4672661234803711,exnex=ex)
                diff=float(np.max(abs(x-y)))
                rows.append(dict(test='collapsed_prior_independent_quad',n=n,method='EXNEX' if ex else 'HB',counts=str([r]),max_abs_difference=diff,passed=diff<a.tolerance))
        print(f'Checked n={n}',flush=True)
    write_csv(out/'checks.csv',rows)
    summary=dict(checks=len(rows),passed=sum(r['passed'] for r in rows),max_abs_difference=max(r['max_abs_difference'] for r in rows),
                 tolerance=a.tolerance,elapsed_seconds=time.monotonic()-t0,
                 limitation='Finite validation cases, not a uniform numerical error bound over every possible dataset or hyperprior.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
    return 0 if all(r['passed'] for r in rows) else 1
if __name__=='__main__':raise SystemExit(main())
