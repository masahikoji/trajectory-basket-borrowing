#!/usr/bin/env python3
"""Regenerate the Excel/CSV/LaTeX reports from complete saved simulation chunks."""
import argparse,json
from pathlib import Path
import numpy as np
from designs import Setting
from simulate import source_hashes
from reporting import summarize

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--no-excel',action='store_true')
    a=p.parse_args();mf=json.loads((a.results/'manifest.json').read_text())
    if mf['hashes']!=source_hashes():raise SystemExit('Current analysis/report source differs from the run. Restore its original package before exporting.')
    es=[]
    for raw in mf['settings']:
        d=raw.copy();d.pop('observation_count_probabilities',None);d.pop('mean_assessments',None)
        for k in ('initial','sequences','labels','early_factors','observation_probabilities'):
            if d[k] is not None:d[k]=np.asarray(d[k])
        e=Setting(**d);e.validate();es.append(e)
    summarize(a.results,es,mf['config'],mf,excel=not a.no_excel)
    print('Regenerated complete-run reports:',a.results)
if __name__=='__main__':main()
