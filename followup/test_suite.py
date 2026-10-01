#!/usr/bin/env python3
import unittest
import numpy as np
from designs import (FOLLOWUP_PROFILES, EXPECTED_MEANS, settings, profile_null,
                     PRIMARY_OBS_PROB, SHORT_OBS_PROB, LONG_OBS_PROB)
from generation import generate_patient_data
from reference.core import BASE_P, BASE_PI, GROUPS, exact_truth, simulate_observed_trial
from reference.designs import plan as original_plan
from analysis import select_partitions

class FollowupDesignTests(unittest.TestCase):
    def test_probabilities_sum_to_one(self):
        for w in FOLLOWUP_PROFILES.values(): self.assertAlmostEqual(float(w.sum()),1.0,places=12)
    def test_means(self):
        for name,w in FOLLOWUP_PROFILES.items():
            self.assertAlmostEqual(float(w@np.arange(1,11)),EXPECTED_MEANS[name],places=12)
    def test_short_values(self):
        np.testing.assert_allclose(SHORT_OBS_PROB,[.03,.25,.25,.25,.10,.10,.02,0,0,0],atol=0,rtol=0)
    def test_long_values(self):
        np.testing.assert_allclose(LONG_OBS_PROB,[.03,.05,.10,.20,.25,.20,.10,.03,.02,.02],atol=1e-15,rtol=0)
    def test_primary_values(self):
        np.testing.assert_allclose(PRIMARY_OBS_PROB,[.03,.05,.25,.25,.20,.10,.05,.03,.02,.02],atol=0,rtol=0)
    def test_default_has_18_alternatives(self): self.assertEqual(len(settings()),18)
    def test_all_profiles_has_27(self): self.assertEqual(len(settings(['primary','short','long'])),27)
    def test_each_profile_has_9(self):
        for p in FOLLOWUP_PROFILES:self.assertEqual(len(settings([p])),9)
    def test_scenario_labels(self):
        for s in (1,2,3):
            e=settings(['short'],[30],[s])[0]
            np.testing.assert_array_equal(e.labels,GROUPS[s])
    def test_profile_p0_values(self):
        self.assertAlmostEqual(profile_null('primary'),.46726612353008795,places=14)
        self.assertAlmostEqual(profile_null('short'),.3971334154843751,places=14)
        self.assertAlmostEqual(profile_null('long'),.5077563536472754,places=14)
    def test_A_is_null_boundary(self):
        for e in settings(['primary','short','long']):self.assertAlmostEqual(e.true_orr[0],e.p0,places=13)
    def test_BC_remain_active(self):
        for p in FOLLOWUP_PROFILES:
            w=FOLLOWUP_PROFILES[p]
            vals=[exact_truth(pi,np.repeat(P[None],9,0),weights=w)['orr'] for pi,P in zip(BASE_PI,BASE_P)]
            self.assertGreater(vals[1],vals[0]);self.assertGreater(vals[2],vals[0])
    def test_common_random_number_stream_key(self):
        for n in (20,30,50):
            for s in (1,2,3):
                keys=[settings([p],[n],[s])[0].stream_key for p in ('primary','short','long')]
                self.assertEqual(len(set(keys)),1)
                self.assertEqual(keys[0],f'main_main_S{s}_n{n}_homogeneous')
    def test_primary_truth_matches_original_main(self):
        ours={(int(e.scenario),e.n):e for e in settings(['primary'])}
        for old in original_plan('main'):
            e=ours[(old.scenario,old.n)]
            np.testing.assert_allclose(e.true_orr,[x['orr'] for x in old.truth],atol=2e-15,rtol=0)
            self.assertAlmostEqual(e.p0,old.p0,places=14)
    def test_primary_generator_matches_reference(self):
        e=settings(['primary'],[30],[2])[0]
        rng=np.random.default_rng(3);u=rng.random((5,30,11))
        a=generate_patient_data(u,e.initial,e.sequences,e.observation_probabilities).aggregate()
        b=simulate_observed_trial(u,e.initial,e.sequences,weights=PRIMARY_OBS_PROB)
        np.testing.assert_array_equal(a.response_counts,b.response_counts)
        np.testing.assert_array_equal(a.initial_counts,b.initial_counts)
        np.testing.assert_array_equal(a.transition_counts,b.transition_counts)
        np.testing.assert_array_equal(a.final_counts,b.final_counts)
        np.testing.assert_array_equal(a.length_counts,b.length_counts)
    def test_short_has_no_lengths_8_to_10(self):
        e=settings(['short'],[50],[1])[0]
        u=np.zeros((5,50,11));u[:,:,0]=.999999
        d=generate_patient_data(u,e.initial,e.sequences,e.observation_probabilities)
        lengths=d.lengths.argmax(-1)+1
        self.assertTrue(np.all(lengths==7))
    def test_analysis_runs_without_truth_inputs(self):
        e=settings(['long'],[20],[3])[0]
        rng=np.random.default_rng(4);u=rng.random((5,20,11))
        d=generate_patient_data(u,e.initial,e.sequences,e.observation_probabilities)
        c,_=select_partitions(d,np.random.default_rng(5),np.random.default_rng(6))
        self.assertEqual(len(c['Proposed']),5)
        self.assertEqual(len(np.unique(c['Proposed'])),len(np.unique(c['Previous'])))
    def test_invalid_profiles_rejected(self):
        with self.assertRaises(ValueError):settings(['short','short'])
    def test_invalid_sizes_rejected(self):
        with self.assertRaises(ValueError):settings(['short'],[2])

if __name__=='__main__':unittest.main(verbosity=2)
