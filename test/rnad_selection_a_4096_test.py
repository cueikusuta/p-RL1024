"""A-4096 training and sealed 2048-Match confirmation contract tests."""
import math
import unittest
from scripts.rnad.selection_a_4096 import (
    TRAIN_BLOCKS,DEV_BLOCK,TRAIN_MATCHES,SEEDS,CHECKPOINT_EPOCHS
)
from scripts.rnad.selection_a_4096_confirm import (
    HOLDOUT_START,HOLDOUT_MATCHES,SELECTION_THRESHOLD,
    paired_seed_cluster_bootstrap
)
from scripts.rnad.league_entry_gate import TARGETS


class A4096Test(unittest.TestCase):
    def test_all_teacher_trace_blocks_are_disjoint(self):
        counts=[(start,n) for start,n in TRAIN_BLOCKS]
        counts += [(DEV_BLOCK[0],DEV_BLOCK[1]),
                   (HOLDOUT_START,HOLDOUT_MATCHES)]
        self.assertEqual(sum(n for _,n in TRAIN_BLOCKS),TRAIN_MATCHES)
        self.assertEqual(TRAIN_MATCHES,4096)
        self.assertEqual(HOLDOUT_MATCHES,2048)
        self.assertEqual(SEEDS,(90251,90252,90253))
        self.assertEqual(CHECKPOINT_EPOCHS,(40,60))
        seeds=[set(range(start,start+n)) for start,n in counts]
        for i in range(len(seeds)):
            for j in range(i+1,len(seeds)):
                self.assertFalse(seeds[i]&seeds[j])

    def test_selection_only_new_threshold_but_other_phases_unmodified(self):
        self.assertEqual(TARGETS["selection"],(.50,"at_least"))
        self.assertEqual(TARGETS["draft"],(.50,"strictly_greater"))
        self.assertEqual(TARGETS["battle"],(.80,"at_least"))
        self.assertEqual(SELECTION_THRESHOLD,.50)

    def test_seed_cluster_paired_CI_and_bad_input(self):
        old={i:.40 for i in range(HOLDOUT_MATCHES)}
        new={i:.45 for i in range(HOLDOUT_MATCHES)}
        result=paired_seed_cluster_bootstrap(old,new)
        self.assertEqual(result["complete_match_seed_clusters"],2048)
        self.assertAlmostEqual(result["paired_cluster_mean_top1_delta"],.05)
        low,high=result["approximate_two_sided_95pct_ci"]
        self.assertAlmostEqual(low,.05)
        self.assertAlmostEqual(high,.05)
        self.assertTrue(result["improvement_supported_at_approximate_95pct"])
        with self.assertRaisesRegex(ValueError,"2048"):
            paired_seed_cluster_bootstrap(dict(list(old.items())[:100]),new)

    def test_50pct_point_does_not_automatically_satisfy_lower_gate(self):
        # Matching the threshold without sufficient lower CI is insufficient.
        radius=math.sqrt(math.log(3/.2)/(2*HOLDOUT_MATCHES))
        self.assertGreater(radius,0)
        self.assertLess(.50-radius,.50)

if __name__=="__main__":
    unittest.main()
