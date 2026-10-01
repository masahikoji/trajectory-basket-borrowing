#!/usr/bin/env python3
"""Analyze five equal-sized baskets from observed categorical response trajectories."""
from __future__ import annotations

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import sys
import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'main'))
from analysis import select_partitions, infer_partitions, validate_observed
from cluster_selection import PatientData
from posterior import FIELDS, Prior
from shared_posterior import prepare_shared_grid, SharedPosteriorEngine

STATES = {'CR': 0, 'PR': 1, 'SD': 2, 'PD': 3}


def load_trajectories(path: Path) -> tuple[PatientData, list[str]]:
    """Reject gaps or inconsistent records instead of treating missing visits as transitions."""
    patients: dict[str, dict[str, dict[int, int]]] = {}
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or set(reader.fieldnames) != {'basket', 'patient', 'assessment', 'state'}:
            raise ValueError('CSV columns must be basket, patient, assessment, state.')
        for line, row in enumerate(reader, start=2):
            if any(row[k] is None for k in reader.fieldnames) or None in row:
                raise ValueError(f'Invalid CSV row at line {line}.')
            basket, patient = row['basket'].strip(), row['patient'].strip()
            state = row['state'].strip().upper()
            if not basket or not patient or state not in STATES:
                raise ValueError(f'Invalid basket, patient, or state at line {line}.')
            try:
                assessment = int(row['assessment'])
            except ValueError as exc:
                raise ValueError(f'Assessment must be an integer at line {line}.') from exc
            if not 1 <= assessment <= 10:
                raise ValueError(f'Assessment must be from 1 to 10 at line {line}.')
            visits = patients.setdefault(basket, {}).setdefault(patient, {})
            if assessment in visits:
                raise ValueError(f'Duplicate assessment at line {line}.')
            visits[assessment] = STATES[state]
    baskets = sorted(patients)
    if len(baskets) != 5:
        raise ValueError('The current implementation requires exactly five baskets.')
    sizes = {len(patients[b]) for b in baskets}
    if len(sizes) != 1 or min(sizes) < 3:
        raise ValueError('Baskets must have equal patient counts, at least three each.')
    n = sizes.pop()
    first = np.zeros((5, n, 4), float)
    final = np.zeros_like(first)
    transitions = np.zeros((5, n, 4, 4), float)
    occupancy = np.zeros_like(first)
    length_counts = np.zeros((5, n, 10), float)
    response = np.zeros((5, n), int)
    for j, basket in enumerate(baskets):
        for i, patient in enumerate(sorted(patients[basket])):
            visits = patients[basket][patient]
            indices = sorted(visits)
            if indices != list(range(1, len(indices) + 1)):
                raise ValueError(f'Nonconsecutive assessments: basket {basket}, patient {patient}.')
            sequence = [visits[k] for k in indices]
            first[j, i, sequence[0]] = 1
            final[j, i, sequence[-1]] = 1
            length_counts[j, i, len(sequence)-1] = 1
            for origin, destination in zip(sequence[:-1], sequence[1:]):
                transitions[j, i, origin, destination] += 1
            occupancy[j, i] = np.bincount(sequence, minlength=4) / len(sequence)
            response[j, i] = any(state < 2 for state in sequence)
    data = PatientData(first, transitions, length_counts, occupancy, final, response)
    validate_observed(data)
    return data, baskets


def analyze(path: Path, p0: float, out: Path, cache: Path, seed: int = 2026,
            level: int = 2) -> None:
    if not np.isfinite(p0) or not 0 < p0 < 1:
        raise ValueError('p0 must be a prespecified null response rate strictly between 0 and 1.')
    if seed < 0:
        raise ValueError('seed must be nonnegative.')
    if out.exists():
        raise FileExistsError(f'Choose a new output directory: {out}')
    data, names = load_trajectories(path)
    order = np.random.default_rng(np.random.SeedSequence([seed, 0])).permutation(5)
    permuted = PatientData(*(getattr(data, field)[order] for field in
                            ('initial','transitions','lengths','occupancy','final','response')))
    choices, _ = select_partitions(permuted,
        np.random.default_rng(np.random.SeedSequence([seed, 1])),
        np.random.default_rng(np.random.SeedSequence([seed, 2])))
    inverse = np.argsort(order)
    choices = {name: labels[inverse] for name, labels in choices.items()}
    prior = Prior()
    print('Preparing posterior quadrature tables...', flush=True)
    directory = prepare_shared_grid(cache, prior=prior, level=level, ns=(data.n,))
    output, labels = infer_partitions(data.response.sum(1), data.n, choices,
                                     ('Proposed',), p0, p0, SharedPosteriorEngine(directory))
    out.mkdir(parents=True)
    with (out / 'basket_summary.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(('basket', 'n', 'responses', 'observed_orr', 'cluster',
                         'posterior_mean', 'lower90', 'upper90', 'prob_gt_null', 'active'))
        for j, name in enumerate(names):
            values = dict(zip(FIELDS, output[0, j]))
            writer.writerow((name, data.n, int(data.response[j].sum()),
                float(data.response[j].mean()), int(labels[0, j])+1,
                values['post_mean'], values['lower90'], values['upper90'],
                values['prob_gt_null'], bool(values['lower90'] > p0)))
    metadata = dict(input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        basket_order=names, n_per_basket=data.n, primary='Hitting_mix50', seed=seed,
        p0=p0, prior=asdict(prior), quad_level=level,
        posterior_conditioning='selected partition',
        frequency_error_calibrated=False,
        python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__)
    (out/'analysis.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(f'Completed: {out / "basket_summary.csv"}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--p0', type=float, required=True)
    parser.add_argument('--out', type=Path, default=Path('results/example'))
    parser.add_argument('--cache', type=Path, default=Path('.cache/example'))
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--quad-level', type=int, choices=(2, 3), default=2)
    args = parser.parse_args()
    try:
        analyze(args.input, args.p0, args.out, args.cache, args.seed, args.quad_level)
    except (ValueError, OSError) as exc:
        parser.exit(1, f'ERROR: {exc}\n')


if __name__ == '__main__':
    main()
