"""I5 A/B 512-Match experiment safeguards and statistic boundaries."""
import unittest
from scripts.rnad.selection_ab_scaling import (
    SEEDS, CHECKPOINT_EPOCHS, TRAIN_BLOCKS, DEV, NEW_MATCHES)
from scripts.rnad.selection_ab_scaling_confirm import (
    paired_bonferroni_two, FRESH_START, FRESH_MATCHES)

class ScalingContractTests(unittest.TestCase):
    def test_all_seed_blocks_are_independent_and_fixed(self):
        self.assertEqual(NEW_MATCHES,512)
        old=set(range(TRAIN_BLOCKS[0][0],
                      TRAIN_BLOCKS[0][0]+TRAIN_BLOCKS[0][1]))
        expanded=set(range(TRAIN_BLOCKS[1][0],
                           TRAIN_BLOCKS[1][0]+TRAIN_BLOCKS[1][1]))
        reused=set(range(DEV[0],DEV[0]+DEV[1]))
        confirmation=set(range(FRESH_START,FRESH_START+FRESH_MATCHES))
        for i,left in enumerate((old,expanded,reused,confirmation)):
            for right in (old,expanded,reused,confirmation)[i+1:]:
                self.assertFalse(left&right)
        self.assertEqual((len(SEEDS),tuple(CHECKPOINT_EPOCHS)),(3,(10,20)))

    def test_bonferroni_uses_512_paired_seed_clusters(self):
        old={i:.2 for i in range(FRESH_MATCHES)}
        better={i:.3 for i in range(FRESH_MATCHES)}
        result=paired_bonferroni_two(old,better)
        self.assertEqual(result["clusters"],512)
        self.assertTrue(result["familywise_95pct_two_comparison_bonferroni"])
        self.assertAlmostEqual(result["mean_match_cluster_difference"],.1)
        self.assertAlmostEqual(
            result["approximate_97_5pct_two_sided_cluster_bootstrap_ci"][0],.1)
        with self.assertRaisesRegex(ValueError,"512"):
            paired_bonferroni_two(old,dict(list(better.items())[:-1]))
        with self.assertRaisesRegex(ValueError,"512"):
            paired_bonferroni_two({1:.2,2:.3},{1:.4,2:.4})

if __name__=="__main__":
    unittest.main()
