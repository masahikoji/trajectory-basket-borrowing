#!/usr/bin/env python3
"""Estimate transition-matrix error in the nine primary simulation settings."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import platform
import sys
import subprocess
import time
import numpy as np
from transition_model import (FIELDS, DEFAULT_SEED, OBS_PROB, main_settings,
                              sample_trial, validate_arrays)

VERSION = '1.0.0'


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def source_hashes():
    root = Path(__file__).resolve().parent
    return {n: file_hash(root/n) for n in ('evaluate.py', 'transition_model.py', 'report.py')}


def prepare_inputs(directories, settings, reps):
    sources = []
    by_name = {}
    for directory in directories:
        root = Path(directory).expanduser().resolve()
        mf = root/'manifest.json'
        if not mf.is_file():
            raise ValueError(f'{root}: manifest.json not found. Alternatively omit --input to regenerate Main trajectories.')
        doc = json.loads(mf.read_text())
        if not str(doc.get('version', '')).startswith('3.0.'):
            raise ValueError(f'{root}: expected an integrated-suite 3.0.x manifest.')
        if doc.get('primary') != 'Proposed=Hitting_mix50':
            raise ValueError(f'{root}: unexpected primary method.')
        config = doc.get('config', {})
        total = int(config.get('reps', 0)); chunk = int(config.get('chunk_size', 0))
        if total < reps or chunk < 1 or 'fingerprint' not in doc:
            raise ValueError(f'{root}: insufficient saved replicates or invalid chunk metadata (requested {reps}).')
        selected = {e.name: e for e in settings}
        matched = []
        for raw in doc.get('settings', []):
            name = raw.get('name')
            if name not in selected:
                continue
            if name in by_name:
                raise ValueError(f'Duplicate source for {name}; supply one completed copy only.')
            e = selected[name]
            checks = [('initial', e.initial), ('sequences', e.sequences),
                      ('observation_count_probabilities', OBS_PROB)]
            if int(raw.get('n', 0)) != e.n:
                raise ValueError(f'{name}: incompatible sample size.')
            for key, value in checks:
                actual = np.asarray(raw.get(key, []), dtype=float)
                if actual.shape != value.shape or not np.allclose(actual, value, atol=1e-13, rtol=0):
                    raise ValueError(f'{name}: {key} differs from the primary Main definition.')
            entries = []
            for start in range(0, reps, chunk):
                stop = min(start+chunk, total)
                path = root/'raw'/f'{name}_r{start:07d}_{stop:07d}.npz'
                if not path.is_file():
                    raise ValueError(f'Missing saved trial chunk: {path}. No partial analysis was performed. '
                                     'Use the complete original folder, or omit --input to regenerate Main trajectories.')
                entries.append(dict(path=str(path), start=start, stop=stop,
                                    size=path.stat().st_size, sha256=file_hash(path)))
            by_name[name] = dict(fingerprint=doc['fingerprint'], entries=entries,
                                 source_seed=config.get('seed'), source_environment=doc.get('environment', {}))
            matched.append(name)
        sources.append(dict(manifest=str(mf), sha256=file_hash(mf), selected_settings=matched))
    missing = [e.name for e in settings if e.name not in by_name]
    if missing:
        raise ValueError('Selected Main settings are missing: ' + ', '.join(missing))
    return by_name, sources


def read_source_window(e, start, stop, source):
    pieces = {k: [] for k in FIELDS}
    for entry in source['entries']:
        a, b = entry['start'], entry['stop']
        if b <= start or a >= stop:
            continue
        file = Path(entry['path'])
        if file_hash(file) != entry['sha256']:
            raise ValueError(f'Input changed during analysis: {file}')
        with np.load(file, allow_pickle=False) as z:
            if str(z['fingerprint']) != source['fingerprint'] or str(z['experiment']) != e.name:
                raise ValueError(f'Incompatible source chunk: {file}')
            if not np.array_equal(z['replicate'], np.arange(a, b)):
                raise ValueError(f'Incomplete or reordered source trials: {file}')
            original = {k: z[k] for k in FIELDS}
            validate_arrays(original, e.n, b-a)
            lo, hi = max(a, start)-a, min(b, stop)-a
            for k in FIELDS:
                pieces[k].append(original[k][lo:hi])
    result = {k: np.concatenate(v) for k, v in pieces.items()}
    validate_arrays(result, e.n, stop-start)
    return result


def chunk_path(out, name, start, stop):
    return Path(out)/'raw'/f'{name}_r{start:07d}_{stop:07d}.npz'


def run_chunk(e, start, stop, seed, source, out, fingerprint):
    if source is None:
        rows = [sample_trial(e, rep, seed) for rep in range(start, stop)]
        arrays = {k: np.stack([row[k] for row in rows]) for k in FIELDS}
    else:
        arrays = read_source_window(e, start, stop, source)
    validate_arrays(arrays, e.n, stop-start)
    path = chunk_path(out, e.name, start, stop)
    temp = path.with_name(path.name+f'.tmp.{os.getpid()}')
    with temp.open('wb') as stream:
        np.savez_compressed(stream, fingerprint=fingerprint, experiment=e.name,
                            replicate=np.arange(start, stop), **arrays)
    os.replace(temp, path)
    return str(path), os.getpid()


def check_chunk(path, e, start, stop, fingerprint):
    with np.load(path, allow_pickle=False) as z:
        if str(z['fingerprint']) != fingerprint or str(z['experiment']) != e.name:
            raise ValueError(f'Incompatible checkpoint: {path}')
        if not np.array_equal(z['replicate'], np.arange(start, stop)):
            raise ValueError(f'Incomplete checkpoint: {path}')
        validate_arrays({k: z[k] for k in FIELDS}, e.n, stop-start)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', nargs='+', type=Path, help='Optional completed integrated-suite results folder(s).')
    p.add_argument('--sizes', nargs='+', type=int, choices=[20,30,50], default=[20,30,50])
    p.add_argument('--scenarios', nargs='+', type=int, choices=[1,2,3], default=[1,2,3])
    p.add_argument('--reps', type=int, default=10000)
    p.add_argument('--seed', type=int, default=None, help=f'Regeneration seed, default {DEFAULT_SEED}. Not used with --input.')
    p.add_argument('--workers', type=int, default=20)
    p.add_argument('--chunk-size', type=int, default=250)
    p.add_argument('--out', type=Path, default=Path('results_transition_precision'))
    p.add_argument('--smoke', action='store_true', help='Use 20 trials per setting; software check only.')
    p.add_argument('--plan-only', action='store_true')
    p.add_argument('--inspect', action='store_true', help='Validate source manifests and required file hashes, without analysis.')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--report-only', action='store_true', help='Regenerate reports from complete own checkpoints; implies resume.')
    p.add_argument('--no-excel', action='store_true', help=argparse.SUPPRESS)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if sys.version_info < (3, 11):
        raise ValueError('Python 3.11 or newer is required.')
    reps = 20 if args.smoke else args.reps
    if reps < 2 or args.workers < 1 or args.chunk_size < 1 or (args.seed is not None and args.seed < 0):
        raise ValueError('Use reps>=2, workers>=1, chunk-size>=1 and a nonnegative seed.')
    if args.input and args.seed is not None:
        raise ValueError('--seed applies only to regeneration; saved input is not changed.')
    if args.inspect and not args.input:
        raise ValueError('--inspect requires --input. Use --plan-only for regeneration.')
    settings = main_settings(args.sizes, args.scenarios)
    mode = 'saved_counts' if args.input else 'regeneration'
    seed = DEFAULT_SEED if args.seed is None else args.seed
    print(f'Main transition precision: {len(settings)} settings, {reps:,} trials per setting; mode={mode}.', flush=True)
    print('Estimator: outgoing-count ratio; empty rows = (0.25,0.25,0.25,0.25).', flush=True)
    print('No cluster selection or ORR posterior calculation is rerun.', flush=True)
    if args.smoke:
        print('SOFTWARE CHECK ONLY: do not use these estimates in the manuscript.', flush=True)
    for e in settings:
        print(f'  {e.name}', flush=True)
    if args.plan_only:
        return 0
    inputs, sources = ({}, [])
    if args.input:
        print('Checking saved counts and source-file hashes...', flush=True)
        inputs, sources = prepare_inputs(args.input, settings, reps)
    if args.inspect:
        print(f'Input plan complete: {len(settings)} selected Main settings. Array contents are validated during extraction.', flush=True)
        return 0
    config = dict(mode=mode, reps=reps, seed=seed if not args.input else None,
                  sizes=args.sizes, scenarios=args.scenarios, smoke=args.smoke,
                  chunk_size=args.chunk_size)
    scientific = dict(version=VERSION, config=config, settings=[e.describe() for e in settings],
                      hashes=source_hashes(), source_manifests=sources, source_chunks=inputs,
                      environment=dict(python=platform.python_version(), numpy=np.__version__),
                      estimator='row count ratio; uniform replacement only for zero row count')
    fingerprint = hashlib.sha256(json.dumps(scientific, sort_keys=True).encode()).hexdigest()
    manifest = dict(scientific, fingerprint=fingerprint)
    out = args.out.expanduser().resolve()
    for source in sources:
        source_root = Path(source['manifest']).parent
        if out == source_root or source_root in out.parents:
            raise ValueError('Choose an output folder outside the original results directory.')
    out.mkdir(parents=True, exist_ok=True)
    lock = out/'.running.lock'
    try:
        fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError:
        raise ValueError(f'Output is locked: {lock}. Do not remove a lock belonging to a running program.')
    with os.fdopen(fd, 'w') as f:
        f.write(f'{platform.node()}:{os.getpid()}\n')
    started = time.monotonic()
    try:
        mf = out/'manifest.json'
        if mf.exists():
            if not (args.resume or args.report_only):
                raise ValueError('Output already exists. Choose a new folder or use --resume.')
            if json.loads(mf.read_text()) != manifest:
                raise ValueError('Resume refused: input files, code, settings, or numerical environment changed.')
        else:
            if args.resume or args.report_only:
                raise ValueError('No existing manifest; remove --resume for a new analysis.')
            if any(p != lock for p in out.iterdir()):
                raise ValueError('Output directory is not empty and has no matching manifest.')
            atomic_json(mf, manifest)
        (out/'raw').mkdir(exist_ok=True)
        tasks = []; completed = 0; total = 0
        for e in settings:
            for start in range(0, reps, args.chunk_size):
                stop = min(start+args.chunk_size, reps); total += 1
                file = chunk_path(out, e.name, start, stop)
                if file.exists():
                    check_chunk(file, e, start, stop, fingerprint); completed += 1
                else:
                    tasks.append((e, start, stop, seed, inputs.get(e.name), str(out), fingerprint))
        if args.report_only and tasks:
            raise ValueError('Report-only requires all checkpoints; use --resume to complete the analysis.')
        atomic_json(out/'execution.json', dict(status='running', requested_trials=reps*len(settings)))
        pids = set()
        if tasks:
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn')) as pool:
                iterator = iter(tasks); pending = set()
                for _ in range(min(2*args.workers, len(tasks))):
                    pending.add(pool.submit(run_chunk, *next(iterator)))
                last_pct = -1
                while pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for job in done:
                        _, pid = job.result(); pids.add(pid); completed += 1
                        task = next(iterator, None)
                        if task is not None:
                            pending.add(pool.submit(run_chunk, *task))
                    pct = int(100*completed/total)
                    if pct//5 > last_pct or completed == total:
                        print(f'Complete chunks {completed}/{total} ({pct}%), elapsed {time.monotonic()-started:.1f}s', flush=True)
                        last_pct = pct//5
        print('Computing estimation errors and CSV summary tables...', flush=True)
        from report import export_reports
        export_reports(out,settings,reps,args.chunk_size,fingerprint,False)
        atomic_json(out/'execution.json', dict(status='completed', mode=mode, settings=len(settings),
                    trials_per_setting=reps, total_trials=reps*len(settings),
                    workers_requested=args.workers, workers_observed=len(pids),
                    elapsed_seconds=time.monotonic()-started, smoke=args.smoke,
                    machine=platform.machine(), python=platform.python_version(), numpy=np.__version__))
        print(f'Completed. Results: {out}', flush=True)
        print('For Excel output, run report.py --results with this output directory.', flush=True)
    finally:
        lock.unlink(missing_ok=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('Interrupted. Completed chunks were retained; rerun the same command with --resume.', file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise SystemExit(1)
