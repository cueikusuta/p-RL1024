"""Versioned R-NaD v4 phase-only loss contract; old profiles stay unchanged."""
import unittest
from scripts.rnad.train import (
    PHASE_LEARNING_PROFILES, phase_eligible_for_loss, train
)


class RnadPhaseProfileTest(unittest.TestCase):
    def test_default_full_phase_profile_unchanged(self):
        self.assertEqual(PHASE_LEARNING_PROFILES['all'],
                         frozenset(('draft','selection','battle')))
        for phase in ('draft','selection','battle'):
            self.assertTrue(phase_eligible_for_loss(phase))
        self.assertFalse(phase_eligible_for_loss('battle','selection-only'))
        self.assertFalse(phase_eligible_for_loss('draft','selection-only'))
        self.assertTrue(phase_eligible_for_loss('selection','selection-only'))

    def test_draft_and_combined_phase_mask(self):
        self.assertTrue(phase_eligible_for_loss('draft','draft-only'))
        self.assertFalse(phase_eligible_for_loss('selection','draft-only'))
        self.assertTrue(phase_eligible_for_loss('selection','draft-selection'))
        self.assertTrue(phase_eligible_for_loss('draft','draft-selection'))
        self.assertFalse(phase_eligible_for_loss('battle','draft-selection'))
        # Legacy low-level synthetic events have no phase; only the
        # unfiltered default policy path may accept them.
        self.assertTrue(phase_eligible_for_loss(None,'all'))
        with self.assertRaises(ValueError):
            phase_eligible_for_loss(None,'selection-only')

    def test_reject_unrecognized_phase_and_non_v4_training(self):
        for phase,profile in [('battle','unknown'),('unknown','all'),
                              ('unknown','selection-only')]:
            with self.subTest(phase=phase,profile=profile):
                with self.assertRaises(ValueError):
                    phase_eligible_for_loss(phase,profile)
        with self.assertRaises(ValueError):
            train(seed=301, matches=1,phase_learning_profile='selection-only')
        with self.assertRaises(ValueError):
            train(seed=301, matches=1,features_v2=True,three_phase=True,
                  phase_learning_profile='selection-only')


if __name__=='__main__':
    unittest.main()
