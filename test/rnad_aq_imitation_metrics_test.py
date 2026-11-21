"""Loss scale / candidate-count adjusted AQ imitation held-out metrics.

No teacher matches, private opponent state or models are required here;
the production evaluator still validates complete trusted sources.
"""
import math
import unittest
import torch
from scripts.rnad.evaluate_aq_imitation import per_decision_metrics


class TeacherAgreementMetricsTest(unittest.TestCase):
    def test_uniform_distribution_compares_60_and_many_legal_candidates(self):
        for n in (1,3,60,2000):
            with self.subTest(n=n):
                r=per_decision_metrics(torch.zeros(n),n-1)
                self.assertAlmostEqual(r['cross_entropy'],math.log(n),places=5)
                self.assertAlmostEqual(r['uniform_cross_entropy'],math.log(n))
                self.assertAlmostEqual(r['log_probability_gain_vs_uniform'],0,places=5)
                self.assertAlmostEqual(r['teacher_action_probability'],1/n,places=6)
                self.assertAlmostEqual(r['uniform_random_top1'],1/n)
                self.assertAlmostEqual(r['uniform_random_top10'],min(10,n)/n)
                self.assertTrue(all(math.isfinite(x) for x in r.values()))

    def test_teacher_rank_topk_and_calibration_change_without_fake_strength(self):
        p=per_decision_metrics(torch.tensor([0.,0.,6.,0.,0.]),2)
        self.assertEqual((p['top1'],p['top3'],p['top10']),(1,1,1))
        self.assertGreater(p['teacher_action_probability'],.90)
        self.assertGreater(p['log_probability_gain_vs_uniform'],0)
        self.assertLess(p['normalized_cross_entropy'],1.0)
        bottom=per_decision_metrics(torch.tensor([0.,0.,6.,0.,0.]),4)
        self.assertEqual(bottom['top1'],0)
        self.assertEqual(bottom['top3'],0)
        self.assertEqual(bottom['top10'],1)
        self.assertGreater(bottom['cross_entropy'],p['cross_entropy'])

    def test_invalid_policy_and_label_rejected(self):
        for logits,label in (
            (torch.tensor([]),0),
            (torch.tensor([float('nan'),0.]),0),
            (torch.tensor([float('inf'),0.]),0),
            (torch.tensor([0.,0.]),-1),
            (torch.tensor([0.,0.]),2),
            (torch.tensor([0.,0.]),True),
            (torch.zeros((2,2)),0)
        ):
            with self.subTest(label=label),self.assertRaises(ValueError):
                per_decision_metrics(logits,label)


if __name__=='__main__':
    unittest.main()
