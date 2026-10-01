"""Input-interface tests against observed patient summaries."""
import csv
import sys
from pathlib import Path
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'examples'))
from analyze_trial import load_trajectories
from analysis import select_partitions
from generation import generate_patient_data
from designs import settings


class InputTests(unittest.TestCase):
    def rows(self):
        return [dict(basket=str(j),patient=str(i),assessment=str(t),state=state)
                for j in range(5) for i in range(3)
                for t,state in ((1,'SD'),(2,'PR'),(3,'PD'))]

    def read(self,rows):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'trial.csv'
            with p.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=['basket','patient','assessment','state'])
                w.writeheader();w.writerows(rows)
            return load_trajectories(p)

    def test_after_response_preserved(self):
        d,_=self.read(self.rows())
        self.assertTrue(np.all(d.transitions[:,:,1,3]==1))
        self.assertTrue(np.all(d.response==1))
        self.assertTrue(np.all(d.final[:,:,3]==1))
    def test_gap_rejected(self):
        rows=self.rows();rows[1]['assessment']='4'
        with self.assertRaises(ValueError):self.read(rows)
    def test_duplicate_rejected(self):
        rows=self.rows();rows.append(rows[0].copy())
        with self.assertRaises(ValueError):self.read(rows)
    def test_missing_state_rejected(self):
        rows=self.rows();rows[0]['state']='NA'
        with self.assertRaises(ValueError):self.read(rows)
    def test_unequal_baskets_rejected(self):
        with self.assertRaises(ValueError):self.read(self.rows()[3:])
    def test_order_invariance(self):
        a,_=self.read(self.rows());b,_=self.read(self.rows()[::-1])
        for k in ('initial','transitions','lengths','occupancy','final','response'):
            np.testing.assert_array_equal(getattr(a,k),getattr(b,k))
    def test_synthetic_example(self):
        d,names=load_trajectories(ROOT/'examples/synthetic_trial.csv')
        self.assertEqual(d.n,20);self.assertEqual(len(names),5)
        z,_=select_partitions(d,np.random.default_rng(3),np.random.default_rng(4))
        self.assertEqual(z['Proposed'].shape,(5,))
    def test_generated_summaries_match_csv(self):
        e=settings('main',sizes=[3],scenarios=[2])[0]
        u=np.random.default_rng(782).random((5,3,11))
        d=generate_patient_data(u,e.initial,e.sequences)
        # Reconstruct the same realized sequences directly from their uniforms.
        rows=[]
        from reference.core import OBS_PROB
        for j in range(5):
            for i in range(3):
                length=np.searchsorted(np.cumsum(OBS_PROB),u[j,i,0],side='right')+1
                s=int(np.sum(u[j,i,1]>=np.cumsum(e.initial[j])))
                for t in range(int(length)):
                    if t:s=int(np.sum(u[j,i,t+1]>=np.cumsum(e.sequences[j,t-1,s])))
                    rows.append(dict(basket=str(j),patient=str(i),assessment=str(t+1),
                                     state=('CR','PR','SD','PD')[s]))
        parsed,_=self.read(rows)
        for k in ('initial','transitions','lengths','occupancy','final','response'):
            np.testing.assert_array_equal(getattr(d,k),getattr(parsed,k))


if __name__=='__main__':unittest.main()
