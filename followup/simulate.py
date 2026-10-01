#!/usr/bin/env python3
"""Run the prespecified simulation settings and export operating characteristics."""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse, hashlib, json, multiprocessing as mp, platform, sys, time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
from dataclasses import asdict
import numpy as np
import scipy
from analysis import ALL_METHODS, DEFAULT_METHODS, WEIGHT_METHODS, select_partitions, infer_partitions
from designs import settings, stable_id
from generation import generate_patient_data
from reference.core import canonical_labels
from posterior import Prior, atomic_json, FIELDS
from shared_posterior import SharedPosteriorEngine as PosteriorEngine, prepare_shared_grid as build_grid

SUITE_VERSION = 'followup-sensitivity-1.0.0'
_ENGINE = None
_STOP = None


def jsonable(x):
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, Path): return str(x)
    raise TypeError(type(x).__name__)


def source_hashes():
    root = Path(__file__).resolve().parent
    names = ('analysis.py','cluster_selection.py','refinement.py','generation.py','designs.py',
             'reference/core.py','reference/designs.py','posterior.py','shared_posterior.py','simulate.py','reporting.py')
    return {n: hashlib.sha256((root/n).read_bytes()).hexdigest() for n in names}


def initialize_worker(directory, stop):
    global _ENGINE, _STOP
    _ENGINE = PosteriorEngine(directory) if directory else None
    _STOP = stop


def sample_trial(e, rep, seed):
    """Identical generating uniforms/order to the prior refinement study.

    Truth labels do not enter this function or any analysis. Randomly permuted
    baskets prevent source-order preferences; outputs are restored afterwards.
    """
    rg = np.random.default_rng(np.random.SeedSequence([seed, stable_id(e.stream_key), rep, 0]))
    order = rg.permutation(5)
    data = generate_patient_data(rg.random((5, e.n, 11)), e.initial[order], e.sequences[order], e.observation_probabilities)
    ra = np.random.default_rng(np.random.SeedSequence([seed, stable_id(e.stream_key), rep, 1]))
    rc = np.random.default_rng(np.random.SeedSequence([seed, stable_id(e.stream_key), rep, 2]))
    choices, diag = select_partitions(data, ra, rc)
    inv = np.argsort(order)
    choices = {k: (z[inv] if k == 'EXNEX' else canonical_labels(z[inv])) for k, z in choices.items()}
    diag = {k: x[inv] for k, x in diag.items()}
    return data, order, choices, diag


def chunk_path(out, e, start, stop):
    return Path(out)/'raw'/f'{e.name}_r{start:07d}_{stop:07d}.npz'


def run_chunk(e, start, stop, config, out, fingerprint):
    B = stop-start; methods = config['methods']; M = len(methods)
    post = np.full((B, M, 5, len(FIELDS)), np.nan)
    labels = np.empty((B, M, 5), np.int8)
    responses = np.empty((B, 5), np.int32)
    initial = np.empty((B, 5, 4), np.int32)
    trans = np.empty((B, 5, 4, 4), np.int32)
    final = np.empty((B, 5, 4), np.int32)
    lengths = np.empty((B, 5, 10), np.int32)
    Phat = np.empty((B, 5, 4, 4)); final_feature = np.empty((B, 5, 4))
    occupancy = np.empty((B, 5, 4)); hit = np.empty((B, 5)); orders = np.empty((B, 5), np.int8)
    patients = {}
    if config['save_patients']:
        for name, shape in [('patient_first', (B,5,e.n)), ('patient_last', (B,5,e.n)),
                            ('patient_length', (B,5,e.n)), ('patient_response', (B,5,e.n)),
                            ('patient_transitions', (B,5,e.n,4,4))]:
            patients[name] = np.empty(shape, np.uint8)
    for a, rep in enumerate(range(start, stop)):
        if _STOP is not None and _STOP.is_set(): raise InterruptedError('Stopped before completing chunk.')
        try:
            data, order, choices, diag = sample_trial(e, rep, config['seed'])
            inv = np.argsort(order); agg = data.aggregate(); r = agg.response_counts[inv]
            if config['cluster_only']:
                labels[a] = np.array([choices[m] for m in methods])
            else:
                post[a], labels[a] = infer_partitions(r, e.n, choices, methods, e.p0, e.legacy_p0, _ENGINE)
            responses[a] = r; orders[a] = order
            initial[a] = agg.initial_counts[inv]; trans[a] = agg.transition_counts[inv]
            final[a] = agg.final_counts[inv]; lengths[a] = agg.length_counts[inv]
            Phat[a] = diag['Phat']; final_feature[a] = diag['final_feature']
            occupancy[a] = diag['occupancy_feature']; hit[a] = diag['fitted_hitting']
            if patients:
                patients['patient_first'][a] = data.initial.argmax(-1)[inv]
                patients['patient_last'][a] = data.final.argmax(-1)[inv]
                patients['patient_length'][a] = (data.lengths.argmax(-1)+1)[inv]
                patients['patient_response'][a] = data.response[inv]
                patients['patient_transitions'][a] = data.transitions[inv]
        except InterruptedError:
            raise
        except Exception as exc:
            raise RuntimeError(f'{e.name}; replicate {rep}: {exc}') from exc
    target = chunk_path(out, e, start, stop)
    temporary = target.with_name(target.name + f'.tmp.{os.getpid()}')
    with temporary.open('wb') as f:
        np.savez_compressed(f, fingerprint=fingerprint, experiment=e.name,
              replicate=np.arange(start, stop), start=start, stop=stop,
              post=post, labels=labels, responses=responses, basket_order=orders,
              initial_counts=initial, transition_counts=trans, final_counts=final,
              length_counts=lengths, Phat=Phat, final_feature=final_feature,
              occupancy_feature=occupancy, fitted_hitting=hit, **patients)
    os.replace(temporary, target)
    return str(target), os.getpid()


def check_chunk(file, fingerprint, start, stop, methods):
    with np.load(file, allow_pickle=False) as z:
        if str(z['fingerprint']) != fingerprint or not np.array_equal(z['replicate'], np.arange(start,stop)):
            raise ValueError(f'Incompatible or incomplete checkpoint: {file}')
        if z['labels'].shape != (stop-start, len(methods), 5) or z['post'].shape != (stop-start,len(methods),5,len(FIELDS)):
            raise ValueError(f'Incorrect checkpoint dimensions: {file}')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profiles', nargs='+', choices=['primary','short','long'], default=['short','long'],
                   help='Follow-up distributions to run. Default: short long. Add primary only for an internal rerun of the main baseline.')
    p.add_argument('--n', type=int, default=None, help='A single sample size per basket.')
    p.add_argument('--sizes', nargs='+', type=int, default=None, help='Sample sizes per basket. Default: 20 30 50.')
    p.add_argument('--scenarios', nargs='+', type=int, choices=[1,2,3], default=[1,2,3])
    p.add_argument('--methods', nargs='+', choices=ALL_METHODS, default=None)
    p.add_argument('--reps', type=int, default=10000)
    p.add_argument('--workers', type=int, default=20)
    p.add_argument('--chunk-size', type=int, default=25)
    p.add_argument('--seed', type=int, default=610202610,
                   help='Uses the same base stream as the primary Main run for common-random-number comparison.')
    p.add_argument('--out', type=Path, default=Path('results_followup_sensitivity'))
    p.add_argument('--cache', type=Path, default=Path('.posterior_cache'))
    p.add_argument('--quad-level', type=int, choices=[1,2,3,4], default=2)
    p.add_argument('--save-patients', action='store_true')
    p.add_argument('--cluster-only', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--aggregate-only', action='store_true')
    p.add_argument('--plan-only', action='store_true')
    p.add_argument('--no-excel', action='store_true')
    return p

def main(argv=None):
    args = parser().parse_args(argv)
    if sys.version_info < (3,11): raise SystemExit('Python 3.11 or newer is required.')
    if min(args.reps, args.workers, args.chunk_size) < 1 or args.reps < 2 or args.seed < 0:
        raise SystemExit('Use reps>=2, workers>=1, chunk-size>=1, seed>=0.')
    if args.n is not None and args.sizes is not None: raise SystemExit('Choose --n OR --sizes, not both.')
    sizes=[args.n] if args.n is not None else (args.sizes or [20,30,50])
    methods = list(args.methods or DEFAULT_METHODS)
    if len(set(methods)) != len(methods): raise SystemExit('Duplicate methods are not allowed.')
    exps = settings(profiles=args.profiles, sizes=sizes, scenarios=args.scenarios)
    if not exps: raise SystemExit('No settings selected.')
    print(f'Frozen primary analysis; follow-up sensitivity only. {len(exps)} settings; methods: {", ".join(methods)}', flush=True)
    for e in exps:
        print(f'{e.name}: n={e.n}, true K={len(np.unique(e.labels))}, mean assessments={e.mean_assessments:.2f}, p0(A)={e.p0:.12f}', flush=True)
    print('Only the observation-count distribution changes. Transition mechanisms and stage-2 prior are unchanged.', flush=True)
    if args.plan_only: return 0
    if args.cluster_only: print('CLUSTER-ONLY DIAGNOSTIC: efficacy outputs will NOT be generated.', flush=True)
    if args.quad_level == 1 and not args.cluster_only: print('WARNING: quad-level 1 is for software smoke checks only.', flush=True)
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True); (out/'raw').mkdir(exist_ok=True)
    config = dict(profiles=args.profiles, n=args.n, sizes=list(sizes), scenarios=args.scenarios,
                  supp_source='not_applicable', threshold_mode='followup_specific_exact_A',
                  methods=methods, reps=args.reps, chunk_size=args.chunk_size, seed=args.seed,
                  quad_level=args.quad_level, save_patients=args.save_patients, cluster_only=args.cluster_only)
    described = json.loads(json.dumps([e.describe() for e in exps], default=jsonable))
    scientific = dict(version=SUITE_VERSION, primary='Proposed=Hitting_mix50', config=config,
                 prior=asdict(Prior()), settings=described, hashes=source_hashes(),
                 environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
                 posterior_fields=list(FIELDS),
                 inference='conditional on estimated partition; mu/tau numerically integrated; no 5% calibration',
                 RNG='common random numbers across primary/short/long for the same scenario,n,rep; only follow-up mapping changes')
    fingerprint = hashlib.sha256(json.dumps(scientific, sort_keys=True).encode()).hexdigest()
    manifest = {**scientific, 'fingerprint': fingerprint}
    lock = out/'.running.lock'
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SystemExit(f'Output locked: {lock}. Do not delete until the earlier process has stopped.')
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    stop = mp.get_context('spawn').Event(); started = time.monotonic(); pids = set(); pool = None
    try:
        mf = out/'manifest.json'
        if mf.exists():
            if not (args.resume or args.aggregate_only): raise SystemExit('Output exists. Use a NEW directory or --resume.')
            if json.loads(mf.read_text()) != manifest: raise SystemExit('Resume refused: configuration, source or library version differs.')
        else:
            if args.resume or args.aggregate_only: raise SystemExit('No existing manifest to resume.')
            if any((out/'raw').iterdir()): raise SystemExit('Raw output exists without a matching manifest; use a clean output directory.')
            atomic_json(mf, manifest)
        tasks = []; finished = 0; total = 0
        for e in exps:
            for start in range(0,args.reps,args.chunk_size):
                end = min(start+args.chunk_size,args.reps); total += 1
                file = chunk_path(out,e,start,end)
                if file.exists():
                    check_chunk(file,fingerprint,start,end,methods); finished += 1
                else: tasks.append((e,start,end,config,str(out),fingerprint))
        if args.aggregate_only and tasks: raise SystemExit('Aggregate-only refused: some required chunks are missing.')
        directory = None
        if tasks and not args.cluster_only:
            args.cache.mkdir(parents=True, exist_ok=True)
            cache_lock = args.cache/'.prepare.lock'
            try: cfd = os.open(cache_lock, os.O_CREAT|os.O_EXCL|os.O_WRONLY)
            except FileExistsError: raise SystemExit('Another process is preparing this posterior cache. Use a different --cache or retry after it finishes.')
            os.write(cfd,str(os.getpid()).encode()); os.close(cfd)
            try:
                print('Preparing/reusing shared posterior integration tables...', flush=True)
                directory = str(build_grid(args.cache, Prior(), args.quad_level, tuple(sorted({e.n for e in exps}))).resolve())
            finally: cache_lock.unlink(missing_ok=True)
        previous = -1
        def progress():
            nonlocal previous
            pct = int(100*finished/total)
            if pct//2 > previous or finished == total:
                print(f'Complete chunks {finished}/{total} ({pct}%), elapsed {time.monotonic()-started:.1f}s', flush=True)
                previous = pct//2
        if tasks:
            if args.workers == 1:
                initialize_worker(directory, stop)
                for task in tasks:
                    _, pid = run_chunk(*task); pids.add(pid); finished += 1; progress()
            else:
                pool = ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn'),
                             initializer=initialize_worker, initargs=(directory,stop))
                # A bounded queue makes Ctrl+C responsive and avoids unbounded submitted work.
                it = iter(tasks); pending = set()
                def submit_next():
                    try: task = next(it)
                    except StopIteration: return False
                    pending.add(pool.submit(run_chunk,*task)); return True
                for _ in range(2*args.workers):
                    if not submit_next(): break
                while pending:
                    ready, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in ready:
                        _, pid = future.result(); pids.add(pid); finished += 1; progress(); submit_next()
                pool.shutdown(wait=True); pool = None
        from reporting import summarize
        print('Writing checked summary tables...', flush=True)
        summarize(out, exps, config, manifest, excel=not args.no_excel)
        atomic_json(out/'execution.json', dict(complete=True, worker_request=args.workers, worker_pids=sorted(pids),
                    resumed=bool(args.resume or args.aggregate_only), elapsed_seconds=time.monotonic()-started,
                    machine=platform.machine(), platform=platform.platform()))
        print(f'Completed. Results: {out}', flush=True)
        return 0
    except KeyboardInterrupt:
        stop.set()
        print('Stopping at replicate boundaries. Completed chunks are retained for --resume.', flush=True)
        return 130
    finally:
        stop.set()
        if pool is not None: pool.shutdown(wait=True, cancel_futures=True)
        lock.unlink(missing_ok=True)

if __name__ == '__main__':
    mp.freeze_support()
    raise SystemExit(main())
