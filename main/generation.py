"""Generate categorical response trajectories and patient-level summaries."""
import numpy as np
from cluster_selection import PatientData
from reference.core import OBS_PROB

def generate_patient_data(uniforms,initial,sequences,weights=OBS_PROB):
    J,n,width=uniforms.shape;Tmax=len(weights)
    if width!=Tmax+1:raise ValueError('Wrong number of independent uniforms.')
    lengths=np.searchsorted(np.cumsum(weights),uniforms[:,:,0],side='right')+1
    states=(uniforms[:,:,1,None]>=np.cumsum(initial,axis=-1)[:,None,:]).sum(-1)
    first=np.eye(4)[states];trans=np.zeros((J,n,4,4));visits=first.copy();ever=states<2
    for l in range(Tmax-1):
        bj,ij=np.nonzero(lengths>=l+2)
        origins=states[bj,ij].copy()
        cdf=np.cumsum(sequences[bj,l,origins,:],axis=-1)
        dest=(uniforms[bj,ij,l+2,None]>=cdf).sum(-1)
        states[bj,ij]=dest;trans[bj,ij,origins,dest]+=1
        visits[bj,ij]+=np.eye(4)[dest];ever[bj,ij]|=dest<2
    return PatientData(first,trans,np.eye(Tmax)[lengths-1],visits/lengths[:,:,None],np.eye(4)[states],ever.astype(int))
