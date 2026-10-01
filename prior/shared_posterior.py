"""Read-only memory-mapped quadrature tables shared across worker processes."""
from pathlib import Path
import os
import numpy as np
from posterior import PosteriorEngine, Prior, build_grid


def prepare_shared_grid(cache_dir, prior=Prior(), level=2, ns=(20,30,50)):
    directory=build_grid(cache_dir,prior,level,ns)
    for n in sorted(set(ns)):
        folder=directory/f'shared_n{n}';folder.mkdir(exist_ok=True)
        with np.load(directory/f'table_n{n}.npz',allow_pickle=False) as z:
            for key in z.files:
                target=folder/f'{key}.npy'
                if target.exists():continue
                temp=target.with_name(target.name+f'.tmp.{os.getpid()}')
                with temp.open('wb') as f:np.save(f,z[key],allow_pickle=False)
                os.replace(temp,target)
    return directory


class SharedPosteriorEngine(PosteriorEngine):
    def table(self,n):
        if n not in self.tables:
            folder=self.directory/f'shared_n{n}'
            names=('like','I','knex','Inex')
            if not all((folder/f'{key}.npy').exists() for key in names):
                raise FileNotFoundError(f'Missing read-only shared tables for n={n}. Run parent preparation.')
            self.tables[n]={key:np.load(folder/f'{key}.npy',mmap_mode='r',allow_pickle=False) for key in names}
        return self.tables[n]
