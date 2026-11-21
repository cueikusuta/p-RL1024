"""Fast correctness tests for bounded three-phase 256-match teacher study."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from scripts.rnad.train_teacher import phase_balanced_mean_loss
from scripts.rnad.train_draft_imitation_256 import (
    TRAIN_SEED,HELDOUT_SEED,TRAIN_MATCHES,HELDOUT_MATCHES,
    cluster_difference,restore_frozen128)
from scripts.rnad.evaluate_aq_imitation import per_decision_metrics

class ThreePhaseImitation256Tests(unittest.TestCase):
    def test_teacher_seed_blocks_disjoint(self):
        train=set(range(TRAIN_SEED,TRAIN_SEED+TRAIN_MATCHES))
        holdout=set(range(HELDOUT_SEED,HELDOUT_SEED+HELDOUT_MATCHES))
        self.assertEqual(len(train),256)
        self.assertEqual(len(holdout),128)
        self.assertFalse(train&holdout)

    def test_selection_and_battle_phase_focus_are_distinct(self):
        events=[
            ('draft',torch.tensor(1.0,requires_grad=True)),
            ('selection',torch.tensor(10.0,requires_grad=True)),
            ('battle',torch.tensor(3.0,requires_grad=True))]
        selection=phase_balanced_mean_loss(events,enabled=True,
                   weights={'draft':1.,'selection':4.,'battle':1.})
        battle=phase_balanced_mean_loss(events,enabled=True,
                   weights={'draft':1.,'selection':1.,'battle':4.})
        self.assertGreater(float(selection),float(battle))
        self.assertAlmostEqual(float(selection),(1+40+3)/6,places=5)
        self.assertAlmostEqual(float(battle),(1+10+12)/6,places=5)
        battle.backward()
        self.assertAlmostEqual(float(events[-1][1].grad),4/6,places=5)

    def test_frozen_model_absence_or_modified_sha_fails_closed(self):
        with tempfile.TemporaryDirectory() as root:
            bad=Path(root)/'model.json'
            bad.write_text(json.dumps({'schema':'rnad-draft-deepsets-weights-v1'}))
            with self.assertRaisesRegex(ValueError,'SHA mismatch'):
                restore_frozen128(bad)

    def test_seed_cluster_imitation_interval_deterministic(self):
        ids=list(range(128))
        b={i:{'top1':0.3} for i in ids}
        c={i:{'top1':0.5} for i in ids}
        first=cluster_difference(b,c,ids,replicates=1200)
        self.assertEqual(first,cluster_difference(b,c,ids,replicates=1200))
        self.assertAlmostEqual(first['mean_diff_top1'],.2)
        lo,hi=first['approximate_familywise_80pct_seed_cluster_CI']
        self.assertAlmostEqual(lo,.2)
        self.assertAlmostEqual(hi,.2)

    def test_battle_job_preserves_utility_trace_schema_with_hard_ce(self):
        # Public AQ traces exported with --utility-labels must be loaded with
        # that schema; mix=0 preserves the original battle hard-label CE.
        workflow=Path('.github/workflows/rnad-i5-threephase-imitation-256.yml').read_text()
        section=workflow.split(
            '- name: Train battle-focused v5 256Match hard-label teacher imitation',
            1)[1].split('- name: Report all three phases',1)[0]
        self.assertIn('--utility-distillation --utility-mix 0',section)
        self.assertIn('--reuse-teacher-traces input/traces/train',section)
        self.assertIn('--phase-profile battle-focus',section)
        self.assertIn('--utility-labels',section)
        # The recover-only workflow must bind the original immutable source.
        recovery=Path('.github/workflows/rnad-i5-threephase-imitation-256-battle-recovery.yml').read_text()
        self.assertIn('022aa9cd5b4ac37514df1d845ea00c11c3c13a5c',recovery)
        self.assertIn('--utility-distillation --utility-mix 0',recovery)

    def test_teacher_imitation_metrics_are_probabilities_not_wins(self):
        v=per_decision_metrics(torch.tensor([0.1,2.0,0.3]),1)
        self.assertEqual(v['top1'],1)
        self.assertEqual(v['top3'],1)
        self.assertGreater(v['teacher_action_probability'],0)
        self.assertLessEqual(v['teacher_action_probability'],1)

if __name__=='__main__':
    unittest.main()
