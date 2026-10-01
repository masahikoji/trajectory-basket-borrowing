"""Generate categorical response trajectories and patient-level summaries."""
import numpy as np
from cluster_selection import PatientData


def generate_patient_data(uniforms,initial,sequences,weights):
    weights=np.asarray(weights,float)
    J,n,width=uniforms.shape;Tmax=len(weights)
    if weights.shape!=(10,) or width!=Tmax+1:raise ValueError('Wrong observation-count distribution or uniforms width.')
    if np.any(weights<0) or not np.isclose(weights.sum(),1.,atol=1e-12,rtol=0):raise ValueError('Invalid observation-count probabilities.')
    lengths=np.searchsorted(np.cumsum(weights),uniforms[:,:,0],side='right')+1
    states=(uniforms[:,:,1,None]>=np.cumsum(initial,axis=-1)[:,None,:]).sum(-1)
    states=np.minimum(states,3)
    first=np.eye(4)[states];trans=np.zeros((J,n,4,4));visits=first.copy();ever=states<2
    for l in range(Tmax-1):
        bj,ij=np.nonzero(lengths>=l+2)
        if not len(bj):break
        origins=states[bj,ij].copy()
        cdf=np.cumsum(sequences[bj,l,origins,:],axis=-1)
        dest=(uniforms[bj,ij,l+2,None]>=cdf).sum(-1);dest=np.minimum(dest,3)
        states[bj,ij]=dest;trans[bj,ij,origins,dest]+=1
        visits[bj,ij]+=np.eye(4)[dest];ever[bj,ij]|=dest<2
    return PatientData(first,trans,np.eye(Tmax)[lengths-1],visits/lengths[:,:,None],np.eye(4)[states],ever.astype(int))
