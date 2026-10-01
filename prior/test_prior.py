"""Software and data-integrity tests. Not a simulation performance study."""
import copy,json,tempfile,unittest
from pathlib import Path
import numpy as np
from prior_io import validate_saved,build_requests,discover,atomic_npz,true_orr,DEFAULT_METHODS
from prior_report import assemble,replay_check,mean_se
from posterior import Prior,FIELDS,independent_table
from dataclasses import asdict

class PriorTests(unittest.TestCase):
    def setUp(self):
        self.methods=list(DEFAULT_METHODS)
        self.r=np.array([[5,9,5,12,9],[3,8,6,2,9]])
        self.z=np.array([[[0,0,1,1,1],[0,1,0,1,1],[0,0,0,0,0],[0,1,2,3,4],[-1]*5]]*2)
        self.e={'n':20,'p0':.4,'legacy_p0':.39}
    def test_saved_valid(self):validate_saved(self.r,self.z,20,self.methods)
    def test_bad_count(self):
        r=self.r.copy();r[0,0]=21
        with self.assertRaises(ValueError):validate_saved(r,self.z,20,self.methods)
    def test_fractional_count(self):
        r=self.r.astype(float);r[0,0]=1.2
        with self.assertRaises(ValueError):validate_saved(r,self.z,20,self.methods)
    def test_wrong_method_axis(self):
        with self.assertRaises(ValueError):validate_saved(self.r,self.z[:,:-1],20,self.methods)
    def test_exnex_hard_partition_rejected(self):
        z=self.z.copy();z[:,-1]=0
        with self.assertRaises(ValueError):validate_saved(self.r,z,20,self.methods)
    def test_one_cluster_checked(self):
        z=self.z.copy();z[:,2,0]=1
        with self.assertRaises(ValueError):validate_saved(self.r,z,20,self.methods)
    def test_independent_checked(self):
        z=self.z.copy();z[:,3]=0
        with self.assertRaises(ValueError):validate_saved(self.r,z,20,self.methods)
    def test_no_input_mutation(self):
        r=self.r.copy();z=self.z.copy()
        build_requests([dict(responses=r,labels=z)],[self.e],self.methods)
        np.testing.assert_array_equal(r,self.r);np.testing.assert_array_equal(z,self.z)
    def test_request_reconstruction(self):
        req,maps=build_requests([dict(responses=self.r,labels=self.z)],[self.e],self.methods)
        for b in range(2):
            for m,method in enumerate(self.methods):
                for j in range(5):
                    k=maps[0]['request_ids'][b,m,j]
                    if method=='Independent':self.assertEqual(k,-1);continue
                    key=req[k];index=maps[0]['positions'][b,m,j]
                    self.assertEqual(key[4][index],self.r[b,j]);self.assertEqual(key[3],method=='EXNEX')
    def test_order_invariance_counts(self):
        req,maps=build_requests([dict(responses=self.r,labels=self.z)],[self.e],self.methods)
        order=np.array([4,2,0,3,1])
        rq,mp=build_requests([dict(responses=self.r[:,order],labels=self.z[:,:,order])],[self.e],self.methods)
        self.assertEqual(req,rq)
    def test_reuse_across_settings(self):
        a=dict(responses=self.r,labels=self.z)
        r1,_=build_requests([a],[self.e],self.methods)
        r2,_=build_requests([a,a],[self.e,self.e],self.methods)
        self.assertEqual(r1,r2)
    def test_distinct_thresholds(self):
        a=dict(responses=self.r,labels=self.z)
        r1,_=build_requests([a],[self.e],self.methods)
        r2,_=build_requests([a,a],[self.e,{**self.e,'p0':.41}],self.methods)
        self.assertEqual(2*len(r1),len(r2))
    def test_independent_unchanged(self):
        a=dict(responses=self.r,labels=self.z)
        req,maps=build_requests([a],[self.e],self.methods)
        lookup=np.zeros((len(req),5,7))
        result=assemble({**a,**maps[0]},self.e,self.methods,lookup)
        expected=independent_table(20,.4,.39)[self.r]
        np.testing.assert_equal(result[:,3],expected)
    def test_replay_failure(self):
        x=np.zeros((2,5,5,7));y=x.copy();y[0,0,0,0]=.01
        rows=replay_check(x,y,.4,self.methods)
        self.assertFalse(rows[0]['passed']);self.assertTrue(rows[1]['passed'])
    def test_replay_unavailable(self):
        x=np.zeros((2,5,5,7));y=np.full_like(x,np.nan)
        self.assertTrue(all(not r['available'] for r in replay_check(x,y,.4,self.methods)))
    def test_paired_se_zero(self):self.assertEqual(mean_se([0,0,0]),(0,0))
    def test_atomic_npz(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'check.npz';atomic_npz(path,responses=self.r)
            with np.load(path) as z:np.testing.assert_array_equal(z['responses'],self.r)
    def test_truth_no_transitions(self):
        pi=np.tile([.1,.2,.3,.4],(5,1));P=np.tile(np.eye(4),(5,9,1,1))
        e=dict(initial=pi,sequences=P,observation_count_probabilities=[.1]*10)
        np.testing.assert_allclose(true_orr(e),.3)
    def test_truth_nonhom(self):
        pi=np.tile([0.,0.,1.,0.],(5,1));P=np.tile(np.eye(4),(5,9,1,1));P[:,1,2]=[0,.5,.5,0]
        e=dict(initial=pi,sequences=P,observation_count_probabilities=[0,0,1,0,0,0,0,0,0,0])
        np.testing.assert_allclose(true_orr(e),.5)
    def test_reject_excel_only(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):discover([d])
    def test_prior_shape_rate(self):
        for rate in (.5,1,2):
            p=Prior(beta=rate);p.validate();self.assertEqual(p.alpha,2);self.assertEqual(p.beta/(p.alpha-1),rate)
    def test_expected_scope(self):
        from prior_io import EXPECTED
        self.assertEqual(len(EXPECTED),12);self.assertNotIn(('supplement','1',20),EXPECTED)

class SourceIntegrityTests(unittest.TestCase):
    fixture=Path(__file__).parent/'validation'/'fixture_source'
    def get_plan(self,path):return discover([path],expected_reps=4)
    def clone(self,tmp):
        import shutil
        dest=Path(tmp)/'source';shutil.copytree(self.fixture,dest);return dest
    def test_full_scope(self):
        ss,es,ms=self.get_plan(self.fixture)
        self.assertEqual(len(es),12);self.assertEqual(ms,list(DEFAULT_METHODS))
    def test_duplicate_source(self):
        with self.assertRaises(ValueError):discover([self.fixture,self.fixture],expected_reps=4)
    def test_nonprimary_source_prior(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=self.clone(tmp);p=d/'manifest.json';m=json.loads(p.read_text());m['prior']['beta']=2;p.write_text(json.dumps(m))
            with self.assertRaises(ValueError):self.get_plan(d)
    def test_missing_chunk(self):
        from prior_io import load_saved
        with tempfile.TemporaryDirectory() as tmp:
            d=self.clone(tmp);ss,es,ms=self.get_plan(d);idx,e=es[0]
            next((d/'raw').glob(e['name']+'_r*.npz')).unlink()
            with self.assertRaises(FileNotFoundError):load_saved(ss[idx],e,ms)
    def test_bad_chunk_fingerprint(self):
        from prior_io import load_saved
        with tempfile.TemporaryDirectory() as tmp:
            d=self.clone(tmp);ss,es,ms=self.get_plan(d);idx,e=es[0]
            p=next((d/'raw').glob(e['name']+'_r*.npz'))
            with np.load(p) as z:a={k:z[k] for k in z.files}
            a['fingerprint']='not-the-source';atomic_npz(p,**a)
            with self.assertRaises(ValueError):load_saved(ss[idx],e,ms)
    def test_bad_replicate_order(self):
        from prior_io import load_saved
        with tempfile.TemporaryDirectory() as tmp:
            d=self.clone(tmp);ss,es,ms=self.get_plan(d);idx,e=es[0]
            p=next((d/'raw').glob(e['name']+'_r*.npz'))
            with np.load(p) as z:a={k:z[k] for k in z.files}
            a['replicate']=a['replicate'][::-1];atomic_npz(p,**a)
            with self.assertRaises(ValueError):load_saved(ss[idx],e,ms)

if __name__=='__main__':unittest.main(verbosity=2)
