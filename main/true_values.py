#!/usr/bin/env python3
"""Exact ORRs for all study settings; no Monte Carlo simulation or posterior fitting."""
import argparse,json
from pathlib import Path
from designs import settings
from reporting import write_csv,status
from simulate import jsonable

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',choices=['all','main','supplement','sensitivity','stress'],default='all')
    p.add_argument('--include-stress',action='store_true')
    p.add_argument('--supp-source',choices=['code','manuscript'],default='code')
    p.add_argument('--out',type=Path,default=Path('true_values.csv'))
    a=p.parse_args();es=settings(a.suite,include_stress=a.include_stress,supp_source=a.supp_source)
    rows=[]
    for e in es:
        for j,v in enumerate(e.true_orr):
            rows.append(dict(setting=e.name,family=e.family,n=e.n,basket=j+1,
                true_K=len(set(e.labels)),true_group=int(e.labels[j])+1,true_ORR=v,
                null_threshold=e.p0,legacy_threshold=e.legacy_p0,efficacy_status=status(v,e.p0)))
    write_csv(a.out,rows)
    a.out.with_suffix('.json').write_text(json.dumps([e.describe() for e in es],default=jsonable,indent=2))
    print('Wrote exact truth table and complete generating specification:',a.out)
if __name__=='__main__':main()
