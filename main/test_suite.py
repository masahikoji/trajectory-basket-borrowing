"""Local software tests. Numerical posterior verification is a separate script."""
import os
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_k]='1'
import copy, hashlib, inspect, json, tempfile, unittest
from pathlib import Path
import numpy as np
from scipy.stats import beta
from designs import settings, stable_id
from analysis import select_partitions, infer_partitions, validate_observed, METHOD_MAP
from cluster_selection import PatientData, fitted_and_jackknife, fitted_distributions, partition_geometry
from refinement import analyze, METHODS, fitted_hitting_probability
from generation import generate_patient_data
from reference.core import canonical_labels, estimated_features, choose_partition
from posterior import independent_table
from reporting import partition_metrics, rate, paired_fields, status
from simulate import sample_trial

SEED=610202610

def data_one(n=8, rep=0):
    e=settings('main',sizes=[n],scenarios=[2])[0]
    rg=np.random.default_rng(12345+rep)
    return generate_patient_data(rg.random((5,n,11)),e.initial,e.sequences)

class TestPlan(unittest.TestCase):
    def test_default_30(self): self.assertEqual(len(settings()),30)
    def test_all_34(self): self.assertEqual(len(settings(include_stress=True)),34)
    def test_main_9(self): self.assertEqual(len(settings('main')),9)
    def test_supplement_3(self): self.assertEqual(len(settings('supplement')),3)
    def test_sensitivity_27(self): self.assertEqual(len(settings('sensitivity')),27)
    def test_stress_4(self): self.assertEqual(len(settings('stress')),4)
    def test_no_duplicate_settings(self):
        ss=settings(include_stress=True); self.assertEqual(len(ss),len({e.name for e in ss}))
    def test_probabilities_valid(self):
        for e in settings(include_stress=True): e.validate()
    def test_time_orr_matched(self):
        es=settings('sensitivity')
        base={(e.scenario,e.n):e for e in es if e.late_odds==1}
        for e in es: np.testing.assert_allclose(e.true_orr,base[(e.scenario,e.n)].true_orr,atol=1e-10,rtol=0)
    def test_exact_reference_values(self):
        self.assertAlmostEqual(settings('main')[0].p0,.467266123530,places=10)
        self.assertAlmostEqual(settings('supplement')[0].p0,.292686926691,places=10)
    def test_supplement_source_not_silently_reconciled(self):
        a=settings('supplement',scenarios=[3])[0]
        b=settings('supplement',scenarios=[3],supp_source='manuscript')[0]
        self.assertFalse(np.array_equal(a.sequences,b.sequences))
        np.testing.assert_allclose(a.true_orr,[.292686926691]*2+[.156914151994]*2+[.595529914344],atol=1e-10)
        np.testing.assert_allclose(b.true_orr,[.413992029029]*2+[.156914151994]*2+[.467266123530],atol=1e-10)
    def test_invalid_size_rejected(self):
        with self.assertRaises(ValueError): settings(sizes=[2])
    def test_stress_homogeneous_not_efficacy_null(self):
        e=settings('stress')[0]; self.assertTrue(np.all(e.true_orr>e.p0))

class TestObserved(unittest.TestCase):
    def test_full_trajectory_counts(self):
        d=data_one();validate_observed(d)
        self.assertTrue(np.any(d.transitions[d.response==1].sum((-1,-2))>1))
    def test_no_generating_arguments(self):
        self.assertEqual(list(inspect.signature(select_partitions).parameters),['data','rng','comparator_rng'])
        self.assertEqual(list(inspect.signature(analyze).parameters),['data','rng'])
    def test_detect_wrong_response(self):
        d=copy.deepcopy(data_one()); d.response[0,0]=1-d.response[0,0]
        with self.assertRaises(ValueError):validate_observed(d)
    def test_detect_noninteger_transition(self):
        d=copy.deepcopy(data_one());d.transitions=d.transitions.astype(float);d.transitions[0,0,0,0]+=.5
        with self.assertRaises(ValueError):validate_observed(d)
    def test_detect_length_disagreement(self):
        d=copy.deepcopy(data_one());d.transitions[0,0,0,0]+=1
        with self.assertRaises(ValueError):validate_observed(d)
    def test_empirical_mles(self):
        d=data_one();a=d.aggregate();_,p,out=estimated_features(a)
        for j in range(5):
            for k in range(4):
                expected=a.transition_counts[j,k]/out[j,k] if out[j,k]>0 else np.full(4,.25)
                np.testing.assert_array_equal(p[j,k],expected)
    def test_uniform_zero_rows(self):
        I=np.tile([0,0,8,0],(5,1));T=np.zeros((5,4,4));L=np.zeros((5,10));L[:,0]=8
        f,o=fitted_distributions(I,T,L)
        np.testing.assert_array_equal(f,I/8);np.testing.assert_array_equal(f,o)
        np.testing.assert_allclose(fitted_hitting_probability(I,T,L),0,atol=1e-15)
    def test_one_observation_fitted_hitting(self):
        I=np.array([2,1,3,4]);T=np.zeros((4,4));L=np.r_[10,np.zeros(9)]
        self.assertAlmostEqual(float(fitted_hitting_probability(I,T,L)),.3)
    def test_hitting_hand_calculation(self):
        I=np.array([0,0,3,0]);T=np.array([[0,0,0,0],[0,0,0,0],[0,1,1,0],[0,0,0,0]])
        L=np.r_[0,0,3,np.zeros(7)]
        self.assertAlmostEqual(float(fitted_hitting_probability(I,T,L)),.75)
    def test_jackknife_recomputes_patient_fit(self):
        d=data_one();n=d.n;f,o,pf,po=fitted_and_jackknife(d)
        j,i=2,3;a=d.aggregate()
        lf,lo=fitted_distributions(a.initial_counts[j]-d.initial[j,i],a.transition_counts[j]-d.transitions[j,i],a.length_counts[j]-d.lengths[j,i])
        np.testing.assert_allclose(pf[j,i],n*f[j]-(n-1)*lf)
        np.testing.assert_allclose(po[j,i],n*o[j]-(n-1)*lo)
    def test_52_partitions_and_15_binary(self):
        _,k,_,_=partition_geometry(30);self.assertEqual(len(k),52);self.assertEqual(int(np.sum(k==2)),15)
    def test_frozen_analyze_identical(self):
        for rep in range(10):
            d=data_one(rep=rep);rng=np.random.default_rng(rep)
            expected,_=analyze(d,rng)
            z,_=select_partitions(d,np.random.default_rng(rep),np.random.default_rng(rep+100))
            for key,name in METHOD_MAP.items():np.testing.assert_array_equal(z[key],expected[METHODS.index(name)])
    def test_k_preservation_and_nontwo_memberships(self):
        for rep in range(40):
            d=data_one(rep=rep);z,_=analyze(d,np.random.default_rng(rep))
            k=[len(np.unique(x)) for x in z];self.assertEqual(len(set(k)),1)
            if k[0]!=2:np.testing.assert_array_equal(z,np.tile(z[0],(len(z),1)))
    def test_original_orr_comparator_unchanged(self):
        d=data_one();z,_=select_partitions(d,np.random.default_rng(2),np.random.default_rng(3))
        expected=choose_partition(d.response.mean(1)[:,None])[0]
        np.testing.assert_array_equal(canonical_labels(z['ORR_only']),canonical_labels(expected))
    def test_source_files_frozen(self):
        root=Path(__file__).resolve().parent;meta=json.loads((root/'provenance/frozen_sources.json').read_text())
        for name,item in meta.items():
            self.assertEqual(hashlib.sha256((root/name).read_bytes()).hexdigest(),item['sha256'],name)

class TestReporting(unittest.TestCase):
    def test_count_not_membership(self):
        z=np.array([[0,0,1,0,1],[0,0,0,1,1]])
        m=partition_metrics(z,np.array([0,0,0,1,1]))
        self.assertTrue(m['correct_number'].all());np.testing.assert_array_equal(m['correct_partition'],[False,True])
    def test_wilson_boundary_not_zero_interval(self):
        s=rate(np.zeros(20));self.assertGreater(s['hi95'],0);self.assertEqual(s['hits'],0)
    def test_paired_difference(self):
        r=paired_fields([1,1,0,0],[0,1,0,0]);self.assertAlmostEqual(r['difference'],.25)
    def test_status_uses_true_response_not_group_number(self):
        self.assertEqual(status(.4,.3),'alternative');self.assertEqual(status(.3,.3),'null_boundary')
        self.assertEqual(status(.2,.3),'null_interior')
    def test_independent_exact_beta(self):
        n=30;r=11;x=independent_table(n,.467,.467)[r]
        np.testing.assert_allclose(x[:3],[(r+1)/(n+2),beta.ppf(.05,r+1,n-r+1),beta.ppf(.95,r+1,n-r+1)])
    def test_bayesian_inference_uses_observed_counts(self):
        class Recorder:
            def __init__(self):self.calls=[]
            def posterior(self,r,n,p,l,exnex=False):
                self.calls.append(np.asarray(r).copy());return independent_table(n,p,l)[r]
        r=np.array([1,3,7,15,20]);rec=Recorder();z={'Proposed':np.array([0,0,0,1,1])}
        ans,_=infer_partitions(r,30,z,['Proposed'],.467,.467,rec)
        np.testing.assert_array_equal(np.concatenate(rec.calls),r)
        np.testing.assert_array_equal(ans[0,:3,:3],independent_table(30,.467,.467)[r[:3],:3])

if __name__=='__main__':unittest.main(verbosity=2)
