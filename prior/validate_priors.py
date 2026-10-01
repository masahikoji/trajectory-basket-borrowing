#!/usr/bin/env python3
"""Finite numerical checks for all three predeclared Gamma shape-rate priors."""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[k]='1'
import argparse,csv,gc,hashlib,json,math,time
from pathlib import Path
import numpy as np
import scipy
from scipy.integrate import quad
from scipy.special import roots_hermitenorm, expit, logit
from scipy.stats import t as student_t, norm
from scipy.optimize import brentq
from posterior import Prior, atomic_json
from shared_posterior import prepare_shared_grid, SharedPosteriorEngine

RATES=(0.5,1.0,2.0)
P0S=(0.467266123530,0.292686926691)


def singleton_reference(r,n,p0,prior=Prior(),exnex=False):
    """Independent integral: integrate tau analytically to Student t, mu by GH."""
    xx,ww=roots_hermitenorm(160);mu=prior.mu0+math.sqrt(prior.sigma0_sq)*xx
    ww=ww/math.sqrt(2*math.pi);scale=math.sqrt(prior.beta/prior.alpha)
    shift=(r*np.log(r/n)+(n-r)*np.log1p(-r/n)) if 0<r<n else 0.
    def density(th):
        pr=float(ww@student_t.pdf((th-mu)/scale,df=2*prior.alpha))/scale
        if exnex:
            pr=prior.ex_probability*pr+(1-prior.ex_probability)*norm.pdf(th,prior.nex_mu,prior.nex_sd)
        return pr*math.exp(r*th-n*np.logaddexp(0,th)-shift)
    def integral(fun,upper):
        edges=[-np.inf]+[v for v in (-30.,-12.,-6.,0.,6.,12.,30.) if v<upper]+[upper]
        return sum(quad(fun,a,b,epsabs=1e-11,epsrel=5e-10,limit=250)[0] for a,b in zip(edges[:-1],edges[1:]))
    Z=integral(density,np.inf)
    mean=integral(lambda th:expit(th)*density(th),np.inf)/Z
    qq=[expit(brentq(lambda th:integral(density,th)/Z-q,-100,100,xtol=1e-10)) for q in (.05,.95)]
    return np.array([mean,*qq,1-integral(density,logit(p0))/Z])


def validation_identity(sizes,rates,level,tolerance):
    root=Path(__file__).parent
    return dict(sizes=list(sizes),rates=list(rates),level=level,tolerance=tolerance,
                numpy=np.__version__,scipy=scipy.__version__,
                hashes={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in ('posterior.py','shared_posterior.py','validate_priors.py')})


def validate(cache,out,sizes=(20,30,50),rates=RATES,level=2,tolerance=2e-5):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    identity=validation_identity(sizes,rates,level,tolerance)
    report=out/'summary.json'
    if report.exists():
        old=json.loads(report.read_text())
        if old.get('identity')==identity and old.get('passed_all'):
            print('Numerical validation already passed for this engine/environment.',flush=True)
            return old
    if level not in (2,3): raise ValueError('Production quadrature level must be 2 or 3.')
    rows=[];start=time.monotonic();rng=np.random.default_rng(61452026)
    for rate in rates:
        prior=Prior(beta=float(rate))
        print(f'Checking Gamma(2, rate={rate}): resolutions {level}/{level+1}...',flush=True)
        lo=SharedPosteriorEngine(prepare_shared_grid(cache,prior,level,sizes))
        hi=SharedPosteriorEngine(prepare_shared_grid(cache,prior,level+1,sizes))
        for n in sizes:
            fixed=[[0],[n],[n//2],[0]*5,[n]*5,[0,1,n-1,n],
                   [round(.46*n)]*3+[round(.64*n)]*2,
                   [round(.29*n)]*2+[round(.16*n)]*2+[round(.60*n)]]
            cases=fixed+[rng.binomial(n,rng.uniform(.05,.95),size=int(rng.integers(1,6))).tolist() for _ in range(3)]
            for ex in (False,True):
                for counts in cases:
                    x=lo.posterior(counts,n,*P0S,exnex=ex)
                    y=hi.posterior(counts,n,*P0S,exnex=ex)
                    columns=7 if ex else 6
                    diff=float(np.max(np.abs(x[:,:columns]-y[:,:columns])))
                    rows.append(dict(test='resolution',rate=rate,n=n,method='EXNEX' if ex else 'HB',
                                     counts=str(counts),max_abs_difference=diff,passed=diff<tolerance))
                for r in (0,n//2,n):
                    x=lo.posterior([r],n,*P0S,exnex=ex)[0]
                    for pidx,p0 in enumerate(P0S):
                        y=singleton_reference(r,n,p0,prior,exnex=ex)
                        actual=x[[0,1,2,3+pidx]]
                        diff=float(np.max(np.abs(actual-y)))
                        rows.append(dict(test='independent_collapsed_singleton',rate=rate,n=n,method='EXNEX' if ex else 'HB',
                                         counts=str([r])+f'; p0={p0}',max_abs_difference=diff,passed=diff<tolerance))
        lo._fit.cache_clear();hi._fit.cache_clear();del lo,hi;gc.collect()
        with (out/'checks.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=dict(identity=identity,checks=len(rows),passed=sum(r['passed'] for r in rows),
                 passed_all=all(r['passed'] for r in rows),max_abs_difference=max(r['max_abs_difference'] for r in rows),
                 elapsed_seconds=time.monotonic()-start,
                 limitation='Finite-case resolution checks and independent singleton calculations; not a uniform error theorem or prior-robustness result.')
    atomic_json(report,summary)
    if not summary['passed_all']: raise ArithmeticError(f'Numerical validation failed. Inspect {out}/checks.csv; do not use results for a manuscript.')
    print(f'Numerical validation passed: {summary["checks"]} checks; max difference {summary["max_abs_difference"]:.3g}',flush=True)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,default=Path('.prior_cache'))
    p.add_argument('--out',type=Path,default=Path('numerical_validation'))
    p.add_argument('--level',type=int,choices=(2,3),default=2)
    a=p.parse_args();validate(a.cache,a.out,level=a.level)

if __name__=='__main__':main()
