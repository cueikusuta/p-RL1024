"""Finite and privacy-preserving phase-candidate collision accounting."""
import copy
import unittest
from unittest.mock import patch

from scripts.rnad.feature_diagnostics import analyze
from scripts.rnad.neural import THREE_PHASE_FEATURE_SCHEMA_V2
from scripts.rnad.train_teacher import AQ_WEIGHTS_SHA256


def accepted():
    rows=[]
    for phase in ('draft','selection','battle'):
        actions=[[.2]*16,[.2]*16,[.3]*16]
        rows.append({'phase':phase,'actor':0,'state':[.1]*32,
                     'actions':actions,'choice':1})
    return {'schema':'rnad-aq-public-belief-three-phase-teacher-v1',
            'feature_schema':THREE_PHASE_FEATURE_SCHEMA_V2,
            'teacher_model_sha256':AQ_WEIGHTS_SHA256,
            'source_sha':'a'*40,'engine_sha':'b'*40,
            'completed':True,'seed':5,'winner':0,'score':[4,2],
            'examples':rows}


class FeatureCollisionTest(unittest.TestCase):
    def test_exact_action_vector_ties_are_counted_per_phase(self):
        with patch('scripts.rnad.feature_diagnostics.current_git_sha',return_value='a'*40), \
             patch('scripts.rnad.feature_diagnostics.frozen_engine_sha',return_value='b'*40):
            report=analyze([accepted()])
            for row in report.values():
                self.assertEqual(row['decisions'],1)
                self.assertEqual(row['legal_candidates'],3)
                self.assertEqual(row['duplicate_candidate_vectors'],1)
                self.assertEqual(row['chosen_has_earlier_identical_vector'],1)
                self.assertEqual(row['chosen_tied_for_identical_vector'],1)
                self.assertAlmostEqual(row['fraction_candidate_vectors_duplicated'],1/3)

    def test_duplicate_seed_secret_schema_and_nonfinite_rejected(self):
        with patch('scripts.rnad.feature_diagnostics.current_git_sha',return_value='a'*40), \
             patch('scripts.rnad.feature_diagnostics.frozen_engine_sha',return_value='b'*40):
            for bad in (
                [accepted(),accepted()],
                [dict(accepted(),feature_schema='wrong')],
                [dict(accepted(),source_sha='wrong')],
                [dict(accepted(),score=[3,2])],
            ):
                with self.subTest(bad=bad[0].get('feature_schema')),self.assertRaises(ValueError):
                    analyze(bad)
            bad=accepted();bad['examples'][0]['actions'][0][0]=float('nan')
            with self.assertRaises(ValueError):
                analyze([bad])
            bad=accepted();bad['examples'][0]['phase']='secret'
            with self.assertRaises(ValueError):
                analyze([bad])


if __name__=='__main__':
    unittest.main()
