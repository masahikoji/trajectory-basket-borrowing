"""Run with: python test_suite.py."""
from __future__ import annotations
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from transition_model import (BASE_P, BASE_PI, OBS_PROB, FIELDS, main_settings,
    estimate_transition, generate_counts, sample_trial, theoretical_support, validate_arrays)
from evaluate import prepare_inputs, read_source_window
from report import estimate_error_stats, row_error_stats, summarize_setting, write_latex

ROOT=Path(__file__).resolve().parent


class PrecisionTests(unittest.TestCase):
    def setUp(self):
        self.e=main_settings([20],[1])[0]
        self.a={k:np.stack([sample_trial(self.e,r)[k] for r in range(4)]) for k in FIELDS}

    def test_01_primary_settings(self):
        plan=main_settings()
        self.assertEqual(len(plan),9)
        self.assertEqual(len({e.name for e in plan}),9)
        for e in plan:self.assertTrue(np.allclose(e.sequences,e.truth[:,None],atol=1e-15,rtol=0))

    def test_02_probability_matrices(self):
        np.testing.assert_allclose(BASE_P.sum(-1),1)
        np.testing.assert_allclose(BASE_PI.sum(-1),1)
        self.assertAlmostEqual(OBS_PROB.sum(),1)
        self.assertAlmostEqual(OBS_PROB@np.arange(1,11),4.45)

    def test_03_zero_rows(self):
        counts=np.zeros((5,4,4))
        np.testing.assert_array_equal(estimate_transition(counts),np.full((5,4,4),.25))

    def test_04_supported_rows(self):
        x=np.arange(16).reshape(4,4)
        np.testing.assert_array_equal(estimate_transition(x),x/x.sum(-1)[:,None])

    def test_05_invalid_estimator_input(self):
        with self.assertRaises(ValueError):estimate_transition(-np.ones((4,4)))
        with self.assertRaises(ValueError):estimate_transition(np.zeros((3,3)))

    def test_06_flow_and_counts(self):
        self.assertEqual(validate_arrays(self.a,20,4),0)

    def test_07_inconsistent_estimates_fail(self):
        a={k:v.copy() for k,v in self.a.items()};a['Phat'][0,0,0,0]+=.1
        with self.assertRaises(ValueError):validate_arrays(a,20)

    def test_08_inconsistent_horizons_fail(self):
        a={k:v.copy() for k,v in self.a.items()};a['length_counts'][0,0,0]+=1
        with self.assertRaises(ValueError):validate_arrays(a,20)

    def test_09_single_assessment(self):
        u=np.full((5,20,11),.1);u[:,:,0]=0
        a=generate_counts(u,self.e.initial,self.e.sequences)
        self.assertEqual(a['transition_counts'].sum(),0)
        np.testing.assert_array_equal(a['Phat'],np.full((5,4,4),.25))
        np.testing.assert_array_equal(a['initial_counts'],a['final_counts'])

    def test_10_post_response_transitions(self):
        u=np.full((5,20,11),.01);u[:,:,0]=.99
        a=generate_counts(u,self.e.initial,self.e.sequences)
        np.testing.assert_array_equal(a['responses'],np.full(5,20))
        self.assertEqual(a['transition_counts'].sum(),5*20*9)

    def test_11_reference_match(self):
        with np.load(ROOT/'validation/reference_trials.npz',allow_pickle=False) as z:
            for i,(s,n,rep) in enumerate(z['cases']):
                e=main_settings([int(n)],[int(s)])[0]
                a=sample_trial(e,int(rep))
                for k in FIELDS:np.testing.assert_array_equal(a[k],z[k][i],err_msg=f'{e.name} r={rep} {k}')

    def test_12_replicate_stream(self):
        for k in FIELDS:np.testing.assert_array_equal(sample_trial(self.e,13)[k],sample_trial(self.e,13)[k])
        self.assertFalse(np.array_equal(sample_trial(self.e,12)['transition_counts'],sample_trial(self.e,13)['transition_counts']))

    def test_13_exact_support_identity_chain(self):
        n=20; pi=BASE_PI[0];x=theoretical_support(pi,np.eye(4),n)
        np.testing.assert_allclose(x['expected_outgoing'],n*3.45*pi)
        np.testing.assert_allclose(x['expected_visits'],n*4.45*pi)
        np.testing.assert_allclose(x['empty_probability'],(1-.97*pi)**n)

    def test_14_total_expected_support(self):
        for pi,P in zip(BASE_PI,BASE_P):
            x=theoretical_support(pi,P,30)
            self.assertAlmostEqual(x['expected_outgoing'].sum(),30*3.45)
            self.assertAlmostEqual(x['expected_visits'].sum(),30*4.45)

    def test_15_error_stats(self):
        x=estimate_error_stats(np.array([.1,.3,.8]),.4)
        self.assertAlmostEqual(x['bias'],0)
        self.assertAlmostEqual(x['mse'],(.09+.01+.16)/3)
        self.assertAlmostEqual(x['rmse']**2,x['mse'])
        self.assertAlmostEqual(x['mcse_bias'],np.std([.1,.3,.8],ddof=1)/np.sqrt(3))

    def test_16_empty_conditional_stats(self):
        x=estimate_error_stats(np.empty(0),.4)
        self.assertTrue(np.isnan(x['rmse']))
        x=row_error_stats(np.empty((0,4)))
        self.assertTrue(np.isnan(x['row_mse']))

    def test_17_row_error_definition(self):
        e=np.array([[.1,-.1,0,0],[.2,-.2,0,0]])
        x=row_error_stats(e)
        self.assertAlmostEqual(x['row_mse'],.0125)
        self.assertAlmostEqual(x['mean_total_variation'],.15)

    def test_18_decomposition(self):
        t=summarize_setting(self.e,self.a)
        self.assertEqual(len(t['cell_errors']),160)
        self.assertEqual(len(t['row_errors']),20)
        self.assertEqual(len(t['support_strata']),100)
        self.assertLess(max(abs(r['decomposition_residual']) for r in t['mse_decomposition']),1e-14)

    def test_19_support_strata_partition(self):
        t=summarize_setting(self.e,self.a)
        for j in range(1,6):
            for r in ('CR','PR','SD','PD'):
                rows=[x for x in t['support_strata'] if x['basket']==j and x['from_state']==r]
                self.assertEqual(sum(x['trials_used'] for x in rows),4)

    def make_source(self,root,mutate=None):
        e=self.e;d=dict(version='3.0.2',primary='Proposed=Hitting_mix50',
            config=dict(reps=4,chunk_size=2,seed=610202610),fingerprint='fixture',
            settings=[e.describe()],environment={'numpy':np.__version__})
        if mutate:mutate(d)
        (root/'manifest.json').write_text(json.dumps(d))
        (root/'raw').mkdir()
        for a,b in [(0,2),(2,4)]:
            np.savez_compressed(root/'raw'/f'{e.name}_r{a:07d}_{b:07d}.npz',
                fingerprint='fixture',experiment=e.name,replicate=np.arange(a,b),
                **{k:v[a:b] for k,v in self.a.items()})

    def test_20_saved_counts_equal_generated(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);self.make_source(root)
            inputs,_=prepare_inputs([root],[self.e],4)
            a=read_source_window(self.e,1,4,inputs[self.e.name])
            for k in FIELDS:np.testing.assert_array_equal(a[k],self.a[k][1:4])

    def test_21_missing_chunk_fails(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);self.make_source(root)
            next((root/'raw').iterdir()).unlink()
            with self.assertRaisesRegex(ValueError,'Missing saved trial chunk'):prepare_inputs([root],[self.e],4)

    def test_22_wrong_main_distribution_fails(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            def mutate(d):d['settings'][0]['observation_count_probabilities']=[.1]*10
            self.make_source(root,mutate)
            with self.assertRaisesRegex(ValueError,'differs'):prepare_inputs([root],[self.e],4)

    def test_23_duplicate_source_fails(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);self.make_source(root)
            with self.assertRaisesRegex(ValueError,'Duplicate'):prepare_inputs([root,root],[self.e],4)

    def test_24_source_changed_fails(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);self.make_source(root)
            inputs,_=prepare_inputs([root],[self.e],4)
            f=Path(inputs[self.e.name]['entries'][0]['path'])
            with f.open('ab') as s:s.write(b'changed')
            with self.assertRaisesRegex(ValueError,'changed'):read_source_window(self.e,0,2,inputs[self.e.name])

    def test_25_all_empty_conditional_denominator(self):
        u=np.full((5,20,11),.1);u[:,:,0]=0
        a=generate_counts(u,self.e.initial,self.e.sequences)
        a={k:np.stack([v,v]) for k,v in a.items()}
        t=summarize_setting(self.e,a)
        for x in t['row_errors']:
            self.assertEqual(x['empty_probability'],1)
            self.assertEqual(x['nonempty_trials'],0)
            self.assertTrue(np.isnan(x['nonempty_row_rmse']))
        self.assertLess(max(abs(r['decomposition_residual']) for r in t['mse_decomposition']),1e-14)

    def test_26_latex_rows(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'table.tex'
            rows=summarize_setting(self.e,self.a)['row_errors']
            write_latex(p,rows)
            entries=[x for x in p.read_text().splitlines() if x.startswith('S1 &')]
            self.assertEqual(len(entries),20)
            self.assertTrue(all(x.endswith('\\\\') for x in entries))


if __name__=='__main__':
    unittest.main(verbosity=2)
