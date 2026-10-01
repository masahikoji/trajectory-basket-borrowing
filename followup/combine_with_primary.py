#!/usr/bin/env python3
"""Combine existing primary Main results with short/long follow-up sensitivity CSVs.

No simulation or recomputation is performed. This is a reporting convenience only.
"""
import argparse,csv
from pathlib import Path


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def write_csv(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def primary_rows(rows):
    out=[]
    for r in rows:
        if r.get('family')!='main' or r.get('dynamics')!='homogeneous':continue
        s=str(r['scenario']); n=str(r['n'])
        q=dict(r);q['setting']=f'followup_primary_S{s}_n{n}';q['family']='followup_sensitivity'
        q['dynamics']='followup_primary';q['followup_profile']='primary';q['mean_assessments']='4.45'
        out.append(q)
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--primary-results',type=Path,required=True,help='Main integrated results_all directory')
    p.add_argument('--sensitivity-results',type=Path,required=True)
    p.add_argument('--out',type=Path,default=Path('combined_followup'))
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    for fn in ('cluster_metrics.csv','basket_metrics.csv','trial_metrics.csv'):
        pp=a.primary_results/fn;ss=a.sensitivity_results/fn
        if not pp.exists() or not ss.exists():raise SystemExit(f'Missing required file: {pp if not pp.exists() else ss}')
        rows=primary_rows(read_csv(pp))+read_csv(ss)
        write_csv(a.out/fn.replace('.csv','_combined.csv'),rows)
    print('Wrote combined Primary/Short/Long tables to',a.out)
if __name__=='__main__':main()
