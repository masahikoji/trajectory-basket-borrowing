#!/usr/bin/env python3
"""Exact ORRs for observation-count sensitivity settings; no Monte Carlo."""
import argparse,json
from pathlib import Path
from designs import settings
from reporting import write_csv,status
from simulate import jsonable

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profiles',nargs='+',choices=['primary','short','long'],default=['primary','short','long'])
    p.add_argument('--sizes',nargs='+',type=int,default=[20,30,50])
    p.add_argument('--scenarios',nargs='+',type=int,choices=[1,2,3],default=[1,2,3])
    p.add_argument('--out',type=Path,default=Path('true_values_followup.csv'))
    a=p.parse_args(); es=settings(a.profiles,a.sizes,a.scenarios)
    rows=[]
    for e in es:
        for j,v in enumerate(e.true_orr):
            rows.append(dict(setting=e.name,followup_profile=e.followup_profile,mean_assessments=e.mean_assessments,
                scenario=e.scenario,n=e.n,basket=j+1,true_K=len(set(e.labels)),true_group=int(e.labels[j])+1,
                true_ORR=v,null_threshold=e.p0,efficacy_status=status(v,e.p0)))
    write_csv(a.out,rows)
    a.out.with_suffix('.json').write_text(json.dumps([e.describe() for e in es],default=jsonable,indent=2))
    print('Wrote exact truth table and complete follow-up sensitivity specification:',a.out)
if __name__=='__main__':main()
