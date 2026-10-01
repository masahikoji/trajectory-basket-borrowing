#!/usr/bin/env python3
"""Optional descriptive figures; no Monte Carlo confidence claim for averaged CIs."""
import argparse,csv
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--methods',nargs='+',default=['Proposed','Previous','ORR_only','One_cluster','Independent','EXNEX'])
    a=p.parse_args()
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:raise SystemExit('Install optional plots: python -m pip install -r requirements-plot.txt')
    with (a.results/'basket_metrics.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    folder=a.results/'figures';folder.mkdir(exist_ok=True)
    for setting in dict.fromkeys(r['setting'] for r in rows):
        rr=[r for r in rows if r['setting']==setting]
        methods=[m for m in a.methods if any(r['method']==m for r in rr)]
        fig,ax=plt.subplots(figsize=(10,5.3))
        offsets=np.linspace(-.3,.3,len(methods)) if len(methods)>1 else [0.]
        truth={int(r['basket']):float(r['true_ORR']) for r in rr}
        for off,m in zip(offsets,methods):
            r=sorted((r for r in rr if r['method']==m),key=lambda r:int(r['basket']))
            x=np.array([int(t['basket']) for t in r])+off
            mean=np.array([float(t['posterior_mean']) for t in r])
            lo=np.array([float(t['mean_lower90']) for t in r]);hi=np.array([float(t['mean_upper90']) for t in r])
            # Values are imported mean endpoints, NOT confidence limits for this simulation mean.
            ax.errorbar(x,mean,yerr=np.array([mean-lo,hi-mean]),fmt='o',capsize=2,label=m)
        ax.plot(sorted(truth),[truth[j] for j in sorted(truth)],'x',label='True ORR',markersize=8)
        ax.axhline(float(rr[0]['p0']),linestyle='--',label='Null response threshold')
        ax.set(xticks=range(1,6),xlabel='Basket',ylabel='ORR',ylim=(0,1),title=setting)
        ax.legend(loc='best',fontsize=8,ncol=2)
        fig.text(.5,.012,'Points: average posterior means. Bars: average 90% posterior interval endpoints.',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.035,1,1));fig.savefig(folder/(setting+'.png'),dpi=170);plt.close(fig)
    print('Figures:',folder)
if __name__=='__main__':main()
