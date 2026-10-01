"""Validate and read saved trial counts, partitions, and posterior summaries."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from posterior import FIELDS, Prior, atomic_json

DEFAULT_METHODS = ('Proposed', 'ORR_only', 'One_cluster', 'Independent', 'EXNEX')
SUPPORTED_METHODS = DEFAULT_METHODS + ('Previous','Corrected_original','Empirical_occupancy',
                    'Refined_w25','Refined_w75','Refined_uniform','Refined_raw50','Empirical_final')
EXPECTED = {('main',str(s),n) for s in (1,2,3) for n in (20,30,50)} | {('supplement',str(s),30) for s in (1,2,3)}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def array_hash(a):
    a = np.ascontiguousarray(a)
    h = hashlib.sha256(str(a.dtype).encode()+repr(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def atomic_npz(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+f'.tmp.{os.getpid()}')
    with temp.open('wb') as f: np.savez_compressed(f, **arrays)
    os.replace(temp,path)


def target_key(e):
    key = (e.get('family'), str(e.get('scenario')), e.get('n'))
    return key if e.get('dynamics') == 'homogeneous' and key in EXPECTED else None


def true_orr(e):
    """Truth is for labeling output only, never passed into posterior inference."""
    pi = np.asarray(e['initial'],float); ps = np.asarray(e['sequences'],float)
    w = np.asarray(e['observation_count_probabilities'],float)
    if pi.shape != (5,4) or ps.shape != (5,9,4,4) or w.shape != (10,):
        raise ValueError('Unrecognized generating-description dimensions in source manifest.')
    for a in (pi,ps,w):
        if np.any(a < 0) or not np.isfinite(a).all() or not np.allclose(a.sum(-1),1,rtol=0,atol=1e-10):
            raise ValueError('Invalid probabilities in source manifest.')
    no_response = pi[:,2:].copy(); ans = w[0]*(1-no_response.sum(-1))
    for t in range(1,10):
        no_response = np.einsum('bi,bij->bj',no_response,ps[:,t-1,2:,2:])
        ans += w[t]*(1-no_response.sum(-1))
    return ans


def validate_saved(responses, labels, n, methods):
    if responses.ndim != 2 or responses.shape[1] != 5:
        raise ValueError('responses must have shape [trials, 5].')
    R = len(responses)
    if labels.shape != (R,len(methods),5):
        raise ValueError('labels shape or source method ordering is inconsistent.')
    if not np.isfinite(responses).all() or np.any(responses != np.floor(responses)) or np.any((responses<0)|(responses>n)):
        raise ValueError('Invalid observed response counts.')
    if not np.isfinite(labels).all() or np.any(labels != np.floor(labels)):
        raise ValueError('Invalid saved partition labels.')
    for m,method in enumerate(methods):
        z = labels[:,m]
        if method == 'EXNEX':
            if not np.all(z == -1): raise ValueError('EXNEX must have no hard partition (label -1).')
        else:
            if np.any((z<0)|(z>4)): raise ValueError(f'Invalid hard partition for {method}.')
            if method == 'One_cluster' and not np.all(z==z[:,:1]):
                raise ValueError('One_cluster has more than one cluster.')
            if method == 'Independent' and any(len(np.unique(x))!=5 for x in z):
                raise ValueError('Independent method contains non-singleton groups.')


def discover(inputs, method_arg=None, expected_reps=10000, allow_subset=False):
    sources=[]; found={}
    for root in inputs:
        root=Path(root).expanduser().resolve()
        if root.is_file() and root.name=='manifest.json': root=root.parent
        if not (root/'manifest.json').is_file():
            raise FileNotFoundError(f'{root}: manifest.json not found. Copy the completed results folder INCLUDING raw/*.npz; an Excel workbook alone is insufficient.')
        if (root/'.running.lock').exists(): raise ValueError(f'Source may still be running: {root}/.running.lock')
        mf=json.loads((root/'manifest.json').read_text()); cfg=mf.get('config',{})
        if not str(mf.get('version','')).startswith('3.') or 'Hitting_mix50' not in mf.get('primary',''):
            raise ValueError('Expected the integrated refined suite 3.x with primary Hitting_mix50; no source model is silently substituted.')
        declared=mf.get('prior')
        if declared is None or Prior(**declared)!=Prior():
            raise ValueError('Source primary prior differs from mu~N(0,1), tau~Gamma(2, rate=1), or the expected EXNEX specification.')
        if mf.get('posterior_fields')!=list(FIELDS): raise ValueError('Unknown posterior-field ordering in manifest.')
        sm=cfg.get('methods',[])
        if len(set(sm))!=len(sm) or any(m not in SUPPORTED_METHODS for m in sm):
            raise ValueError('Unsupported or duplicate source methods.')
        if cfg.get('reps')!=expected_reps:
            raise ValueError(f'{root}: source has {cfg.get("reps")} replicates, expected {expected_reps}. Do not use smoke-check data for manuscript results.')
        if int(cfg.get('chunk_size',0))<1 or not mf.get('fingerprint'):
            raise ValueError('Missing chunk size or source fingerprint.')
        sources.append(dict(root=str(root),manifest_sha256=sha256(root/'manifest.json'),manifest=mf))
        for e in mf.get('settings',[]):
            key=target_key(e)
            if key is None: continue
            name=e.get('name','')
            if not name or Path(name).name!=name: raise ValueError('Unsafe or empty setting name.')
            if key in found: raise ValueError(f'Duplicate source setting {key}. Supply only one authoritative copy of each setting.')
            for t in ('p0','legacy_p0'):
                if not 0<float(e.get(t,-1))<1: raise ValueError('Invalid saved decision threshold.')
            true_orr(e)
            found[key]=(len(sources)-1,e)
    if not found: raise ValueError('No eligible main/supplement homogeneous settings found.')
    missing=EXPECTED-set(found)
    if missing and not allow_subset: raise ValueError(f'Missing requested settings: {sorted(missing)}. Use the original results_all folder, not the time-sensitivity-only folder.')
    available=set.intersection(*(set(s['manifest']['config']['methods']) for s in sources))
    methods=list(DEFAULT_METHODS if method_arg is None else (sorted(available) if method_arg==['all'] else method_arg))
    if not methods or len(set(methods))!=len(methods) or any(m not in available for m in methods):
        raise ValueError(f'Selected methods not present in every source. Available: {sorted(available)}')
    if 'Proposed' not in methods: raise ValueError('Include Proposed in the prior sensitivity.')
    ordered=sorted(found.items(),key=lambda kv:(kv[0][0]!='main',kv[0][2],kv[0][1]))
    return sources,[(idx,e) for _,(idx,e) in ordered],methods


def load_saved(source,e,methods,limit=None):
    """Check every source chunk, even when only a few rows are used for a smoke test."""
    root=Path(source['root']); mf=source['manifest']; cfg=mf['config']; R=cfg['reps']
    indices=[cfg['methods'].index(m) for m in methods]
    B=cfg['chunk_size']; blocks=[]; inventory=[]
    count=R if limit is None else min(int(limit),R)
    if count<2: raise ValueError('At least two trials are required.')
    for start in range(0,R,B):
        stop=min(start+B,R); file=root/'raw'/f'{e["name"]}_r{start:07d}_{stop:07d}.npz'
        if not file.is_file(): raise FileNotFoundError(f'Missing saved trial chunk: {file}')
        with np.load(file,allow_pickle=False) as z:
            if str(z['fingerprint'])!=mf['fingerprint'] or str(z['experiment'])!=e['name']:
                raise ValueError(f'Source fingerprint/setting mismatch: {file}')
            if not np.array_equal(z['replicate'],np.arange(start,stop)):
                raise ValueError(f'Missing, duplicated or reordered replicate IDs: {file}')
            r=z['responses']; lab=z['labels']
            validate_saved(r,lab,e['n'],cfg['methods'])
            if start<count:
                take=min(stop,count)-start
                post=z['post']
                if post.shape!=(stop-start,len(cfg['methods']),5,len(FIELDS)):
                    raise ValueError(f'Incorrect stored posterior dimensions: {file}')
                blocks.append((r[:take],lab[:take,indices],post[:take,indices]))
        inventory.append(dict(path=str(file),sha256=sha256(file)))
    arr=dict(responses=np.concatenate([b[0] for b in blocks]),
             labels=np.concatenate([b[1] for b in blocks]),
             source_post=np.concatenate([b[2] for b in blocks]),replicate=np.arange(count))
    if len(arr['responses'])!=count: raise AssertionError('Internal replica-count mismatch.')
    return arr,inventory


def build_requests(arrays, settings, methods):
    """Deduplicate exact posterior problems without averaging response data.

    Maps are built in one pass to avoid retaining millions of Python objects.
    A final stable sort makes request order independent of source chunking.
    """
    lookup={};mappings=[]
    for a,e in zip(arrays,settings):
        ids=np.full(a['labels'].shape,-1,np.int32)
        positions=np.zeros(a['labels'].shape,np.int8)
        n=int(e['n']);p0=float(e['p0']);old=float(e['legacy_p0'])
        for b,(r,zs) in enumerate(zip(a['responses'],a['labels'])):
            for m,method in enumerate(methods):
                if method=='Independent':continue
                groups=[np.arange(5)] if method=='EXNEX' else [np.flatnonzero(zs[m]==g) for g in np.unique(zs[m])]
                for idx in groups:
                    order=np.argsort(r[idx],kind='stable');inv=np.argsort(order)
                    counts=tuple(int(v) for v in r[idx][order])
                    key=(n,p0,old,method=='EXNEX',counts)
                    if key not in lookup:lookup[key]=len(lookup)
                    ids[b,m,idx]=lookup[key];positions[b,m,idx]=inv
        mappings.append(dict(request_ids=ids,positions=positions))
    requests=sorted(lookup)
    remap=np.empty(len(requests),np.int32)
    for new,key in enumerate(requests):remap[lookup[key]]=new
    for mapping in mappings:
        ids=mapping['request_ids'];valid=ids>=0;ids[valid]=remap[ids[valid]]
    return requests,mappings
