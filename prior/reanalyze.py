#!/usr/bin/env python3
"""Recalculate conditional posterior summaries under alternative precision priors."""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[k]='1'
import argparse,hashlib,json,multiprocessing as mp,platform,sys,time
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path
import numpy as np
import scipy
from prior_io import discover,load_saved,build_requests,atomic_npz,array_hash,true_orr,sha256
from posterior import Prior,FIELDS,atomic_json
from shared_posterior import prepare_shared_grid,SharedPosteriorEngine
from validate_priors import validate,RATES

VERSION='1.0.0'
_ENGINE=None
_STOP=None


def worker_init(directory,stop):
    global _ENGINE,_STOP
    _ENGINE=SharedPosteriorEngine(directory);_STOP=stop


def fit_block(start,requests,path,fingerprint):
    ans=np.full((len(requests),5,len(FIELDS)),np.nan)
    for i,key in enumerate(requests):
        if _STOP is not None and _STOP.is_set():raise InterruptedError('Stopping at request boundary.')
        n,p0,old,ex,counts=key
        try:ans[i,:len(counts)]=_ENGINE.posterior(counts,n,p0,old,exnex=ex)
        except Exception as exc:raise RuntimeError(f'Posterior request {start+i}, n={n}, counts={counts}, EXNEX={ex}: {exc}') from exc
    atomic_npz(path,post=ans,request_ids=np.arange(start,start+len(requests)),fingerprint=fingerprint)
    return os.getpid()


def check_block(path,start,stop,requests,fingerprint):
    with np.load(path,allow_pickle=False) as z:
        if str(z['fingerprint'])!=fingerprint or not np.array_equal(z['request_ids'],np.arange(start,stop)):
            raise ValueError(f'Incompatible posterior checkpoint: {path}')
        arr=z['post']
        if arr.shape!=(stop-start,5,len(FIELDS)):raise ValueError('Incorrect posterior checkpoint dimensions.')
        for a,key in zip(arr,requests):
            if not np.isfinite(a[:len(key[4]),:6]).all():raise ValueError('Nonfinite posterior checkpoint.')
        return arr


def calculate_rate(requests,rate,out,cache,level,workers,blocksize,fingerprint):
    folder=out/'fits'/f'rate{rate:g}';folder.mkdir(parents=True,exist_ok=True)
    paths=[];pending_tasks=[];done=0;total=len(requests)
    for start in range(0,total,blocksize):
        stop=min(start+blocksize,total);path=folder/f'block_{start:07d}_{stop:07d}.npz';paths.append((start,stop,path))
        if path.exists():check_block(path,start,stop,requests[start:stop],fingerprint);done+=stop-start
        else:pending_tasks.append((start,requests[start:stop],str(path),fingerprint))
    if pending_tasks:
        print(f'Gamma(2, rate={rate:g}): preparing/reusing integration grid.',flush=True)
        grid=prepare_shared_grid(cache,Prior(beta=rate),level,sorted({k[0] for k in requests}))
        stop_event=mp.get_context('spawn').Event();pool=None;t0=time.monotonic();pids=set();last=-1
        def progress():
            nonlocal last
            pct=int(100*done/total)
            if pct//5!=last or done==total:
                print(f'  Rate {rate:g}: {done}/{total} unique posterior requests ({pct}%), elapsed {time.monotonic()-t0:.1f}s',flush=True);last=pct//5
        try:
            if workers==1:
                worker_init(str(grid),stop_event)
                for task in pending_tasks:pids.add(fit_block(*task));done+=len(task[1]);progress()
            else:
                pool=ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=worker_init,initargs=(str(grid),stop_event))
                it=iter(pending_tasks);pending={}
                def submit():
                    try:t=next(it)
                    except StopIteration:return False
                    pending[pool.submit(fit_block,*t)]=len(t[1]);return True
                for _ in range(2*workers):
                    if not submit():break
                while pending:
                    ready,_=wait(pending,return_when=FIRST_COMPLETED)
                    for f in ready:
                        done+=pending.pop(f);pids.add(f.result());progress();submit()
        finally:
            stop_event.set()
            if pool is not None:pool.shutdown(wait=True,cancel_futures=True)
        execution=dict(rate=rate,workers_requested=workers,worker_pids=sorted(pids),completed=True)
        atomic_json(folder/'execution.json',execution)
    lookup=np.full((total,5,len(FIELDS)),np.nan)
    for start,stop,path in paths:lookup[start:stop]=check_block(path,start,stop,requests[start:stop],fingerprint)
    return lookup


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,nargs='+',required=True,help='Completed source result folders containing manifest.json and raw/*.npz; never Excel alone.')
    p.add_argument('--out',type=Path,default=Path('results_prior'))
    p.add_argument('--workers',type=int,default=6,help='Default 6 processes for M1 Max; each BLAS process uses one thread.')
    p.add_argument('--methods',nargs='+',default=None,help='Default: Proposed ORR_only One_cluster Independent EXNEX. all uses all methods saved in the source.')
    p.add_argument('--expected-reps',type=int,default=10000,help='Expected number of saved trials in every setting (default 10000). No new trials are generated.')
    p.add_argument('--smoke',action='store_true',help='Use first 20 saved trials per setting; no numerical validation; outputs prominently labeled not for manuscript.')
    p.add_argument('--inspect',action='store_true',help='Validate source availability/counts/partitions and print plan without posterior computations.')
    p.add_argument('--allow-subset',action='store_true',help='Software tests only; requires --smoke. Production requires all 12 settings.')
    p.add_argument('--cache',type=Path,default=Path('.prior_cache'))
    p.add_argument('--quad-level',type=int,choices=(2,3),default=2)
    p.add_argument('--block-size',type=int,default=32)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--no-excel',action='store_true')
    return p


def main(argv=None):
    a=parser().parse_args(argv)
    if sys.version_info<(3,11):raise SystemExit('Python >=3.11 is required. Do not downgrade NumPy to use Python 3.7.')
    if min(a.workers,a.block_size)<1 or a.expected_reps<2:raise SystemExit('Invalid worker/block/trial counts.')
    if a.allow_subset and not a.smoke:raise SystemExit('--allow-subset is only for --smoke tests; production requires all 12 requested settings.')
    if platform.system()=='Darwin' and platform.machine()!='arm64':
        print('WARNING: Intel/Rosetta Python detected. Prefer native arm64 Python on Apple Silicon.',flush=True)
    if a.workers>8: print('WARNING: more than 8 workers is not recommended as the default for M1 Max.',flush=True)
    sources,entries,methods=discover(a.input,a.methods,a.expected_reps,a.allow_subset)
    arrays=[];settings=[];files=[]
    for idx,e in entries:
        data,inventory=load_saved(sources[idx],e,methods,20 if a.smoke else None)
        arrays.append(data);files+=inventory
        row={k:e[k] for k in ('name','family','scenario','n','p0','legacy_p0','dynamics','source')}
        row.update(true_orr=true_orr(e).tolist(),source_root=sources[idx]['root'],
                   response_hash=array_hash(data['responses']),partition_hash=array_hash(data['labels']))
        settings.append(row)
        print(f'Checked {e["name"]}: saved={a.expected_reps}, used={len(data["responses"])}, p0={e["p0"]:.12f}',flush=True)
    print(f'{len(settings)} generating settings x 3 priors; methods: {", ".join(methods)}. Counts, partitions and thresholds remain fixed.',flush=True)
    if a.inspect:return 0
    smoke=a.smoke or a.expected_reps!=10000 or len(settings)!=12
    requests,mappings=build_requests(arrays,settings,methods)
    print(f'Exact reuse: {len(requests)} distinct HB/EXNEX posterior problems per prior.',flush=True)
    root=Path(__file__).resolve().parent
    names=('reanalyze.py','prior_io.py','prior_report.py','posterior.py','shared_posterior.py','validate_priors.py')
    science=dict(version=VERSION,primary='Proposed=Hitting_mix50',settings=settings,methods=methods,rates=list(RATES),
                 used_reps=len(arrays[0]['responses']),expected_source_reps=a.expected_reps,smoke=smoke,
                 source_manifests=[{'root':s['root'],'sha256':s['manifest_sha256'],'fingerprint':s['manifest']['fingerprint'],
                                    'version':s['manifest']['version'],'environment':s['manifest']['environment']} for s in sources],
                 input_files=files,requests=requests,block_size=a.block_size,quad_level=a.quad_level,
                 hashes={n:sha256(root/n) for n in names},environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__),
                 unchanged=dict(mu_prior='N(0,1)',shape=2,EX_probability=.5,NEX_prior='N(0,1)',Independent='Beta(1,1)',
                                counts=True,partitions=True,thresholds=True),
                 inference='Saved-partition conditional Bayesian analysis; mu and tau marginalized, not empirical Bayes. No cutoff calibration.')
    science=json.loads(json.dumps(science))
    fingerprint=hashlib.sha256(json.dumps(science,sort_keys=True).encode()).hexdigest();manifest={**science,'fingerprint':fingerprint}
    out=a.out.expanduser().resolve();cache=a.cache.expanduser().resolve()
    if any(out==Path(s['root']) or Path(s['root']) in out.parents for s in sources):
        raise SystemExit('Use an output folder outside every source results folder; source files are read-only.')
    out.mkdir(parents=True,exist_ok=True);cache.mkdir(parents=True,exist_ok=True)
    lock=out/'.running.lock';cachelock=cache/'.replay.lock'
    for path in (lock,cachelock):
        if path.exists():raise SystemExit(f'Locked: {path}. Do not remove a lock until the corresponding earlier process has stopped.')
    handles=[]
    try:
        for path in (lock,cachelock):
            fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd);handles.append(path)
        mf=out/'manifest.json'
        if mf.exists():
            if not a.resume:raise SystemExit('Output already exists. Use --resume with unchanged arguments, or a new --out directory.')
            if json.loads(mf.read_text())!=manifest:raise SystemExit('Resume refused: input arrays/files, settings, code, numerical versions or options differ.')
        else:
            if a.resume:raise SystemExit('No manifest to resume.')
            if any(p.name not in ('.running.lock',) for p in out.iterdir()):raise SystemExit('Output is not empty. Choose a clean --out directory.')
            atomic_json(mf,manifest)
        (out/'inputs').mkdir(exist_ok=True)
        for data,e,mapping in zip(arrays,settings,mappings):
            dest=out/'inputs'/f'{e["name"]}.npz'
            if not dest.exists():atomic_npz(dest,**data,**mapping)
            else:
                with np.load(dest,allow_pickle=False) as z:
                    if any(not np.array_equal(z[k],v,equal_nan=True) for k,v in {**data,**mapping}.items()):
                        raise ValueError('Copied saved trials or request mappings changed; resume stopped.')
        del arrays,mappings
        t0=time.monotonic()
        if not a.smoke:
            validate(cache,out/'numerical_validation',level=a.quad_level)
        else:print('SMOKE CHECK: numerical validation skipped; these results are NOT for a manuscript.',flush=True)
        lookups={}
        for rate in RATES:
            lookups[rate]=calculate_rate(requests,rate,out,cache,a.quad_level,a.workers,a.block_size,fingerprint)
        from prior_report import summarize
        replay=summarize(out,manifest,lookups,excel=not a.no_excel)
        atomic_json(out/'execution.json',dict(complete=True,workers=a.workers,machine=platform.machine(),
                    platform=platform.platform(),smoke=smoke,elapsed_seconds=time.monotonic()-t0,primary_replay=replay))
        print(f'Completed. Results: {out}',flush=True)
        if smoke:print('SMOKE/SOFTWARE TEST ONLY - do not use these estimates in the manuscript.',flush=True)
        return 0
    finally:
        for path in reversed(handles):path.unlink(missing_ok=True)

if __name__=='__main__':
    mp.freeze_support()
    try:raise SystemExit(main())
    except KeyboardInterrupt:
        print('Interrupted. Completed posterior request blocks are retained; rerun with --resume.',file=sys.stderr);raise SystemExit(130)
    except (ValueError,FileNotFoundError,ArithmeticError) as exc:
        print(f'ERROR: {exc}',file=sys.stderr);raise SystemExit(2)
