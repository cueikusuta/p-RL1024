"""Scaling study contracts: independent Match-seeds and false-positive controls."""
from __future__ import annotations
import copy
import unittest
from unittest.mock import patch
from scripts.rnad.draft_battle_learning_512 import (
    SOURCE, NEW_START, NEW_COUNT, HOLDOUT_START, HOLDOUT_COUNT,
    verified_plan, phase_summary, paired_bootstrap
)


class DraftBattleScaling512Test(unittest.TestCase):
    def test_seed_plan_and_battle_warmstart_are_separate(self):
        with patch("scripts.rnad.draft_battle_learning_512.current_git_sha",
                   return_value=SOURCE), patch(
                "scripts.rnad.draft_battle_learning_512.frozen_engine_sha",
                return_value="b" * 64):
            p = verified_plan()
        self.assertEqual(p["new_draft_training_complete_matches"], 512)
        self.assertEqual(p["new_training"], [NEW_START, NEW_COUNT])
        self.assertEqual(p["unseen_heldout"], [HOLDOUT_START, HOLDOUT_COUNT])
        self.assertEqual(p["old_training"], [2471001, 256])
        self.assertEqual(p["old_used_development"], [2472001, 128])
        self.assertNotEqual(p["new_training"], p["unseen_heldout"])
        self.assertIn("new_optimizer", p["battle_training_sequence"])

    def test_95pct_cluster_bootstrap_is_seed_paired_deterministic(self):
        vals = [.125] * HOLDOUT_COUNT
        self.assertEqual(paired_bootstrap(vals, count=1200), [.125, .125])
        with self.assertRaisesRegex(ValueError, "independent"):
            paired_bootstrap(vals[:-1], count=1200)

    def test_threshold_bound_differs_from_point_estimate(self):
        rows = [
            {"seed": HOLDOUT_START + i, "count": 2, "old": 1, "new": 1}
            for i in range(HOLDOUT_COUNT)
        ]
        base = phase_summary(rows, .50, "at_least")
        self.assertEqual(base["new_decision_weighted_top1"], .5)
        self.assertFalse(base["individual_phase_imitations_target_reached"])
        self.assertFalse(base["learning_improvement_supported_at_95pct"])
        for x in rows:
            x["new"] = 2
        improved = phase_summary(rows, .8, "at_least")
        self.assertEqual(improved["new_decision_weighted_top1"], 1.)
        self.assertGreater(improved["paired_delta_seed_cluster_mean"], 0.)
        self.assertEqual(
            improved["paired_delta_approximate_two_sided_95pct_cluster_bootstrap_CI"],
            [.5, .5])
        self.assertTrue(improved["individual_phase_imitations_target_reached"])

    def test_reject_duplicate_or_missing_seed_rows(self):
        rows = [
            {"seed": HOLDOUT_START + i, "count": 1, "old": 0, "new": 0}
            for i in range(HOLDOUT_COUNT)
        ]
        bad = copy.deepcopy(rows)
        bad[-1]["seed"] = bad[0]["seed"]
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            phase_summary(bad, .5, "strictly_greater")
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            phase_summary(rows[:-1], .5, "strictly_greater")

    def test_absence_of_frozen_git_source_fails_closed(self):
        with patch("scripts.rnad.draft_battle_learning_512.current_git_sha",
                   return_value="e" * 40), self.assertRaisesRegex(
                       ValueError, "exact frozen"):
            verified_plan()


if __name__ == "__main__":
    unittest.main()
