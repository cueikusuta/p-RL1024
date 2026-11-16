"""Fast invariance/gradient checks for separate draft-team research model."""
import unittest
import torch
from scripts.rnad.train_draft_deepsets import (
    DeepSetsDraft, AdditiveDraft, FEATURE_WIDTH,
    teacher_soft_loss, mon_features)

class DeepSetsDraftTest(unittest.TestCase):
    def test_feature_schema(self):
        x={"base_stats":{k:100 for k in ("hp","atk","def","spa","spd","spe")},
           "types":["Fire","Flying"],"roles":["screen","setup"],
           "credit_cost":5,"mega":False,"is_carry":True}
        self.assertEqual(len(mon_features(x)), FEATURE_WIDTH)

    def test_both_models_are_permutation_invariant(self):
        torch.manual_seed(11)
        pool=torch.rand(10,FEATURE_WIDTH)
        choices=torch.tensor([[0,1,2,3,4,5],[1,3,4,5,6,7]])
        permuted=choices[:,[4,0,2,5,1,3]]
        extra=torch.rand(2,4)
        for model in (AdditiveDraft(),DeepSetsDraft()):
            a=model(pool,choices,extra)
            b=model(pool,permuted,extra)
            torch.testing.assert_close(a,b,rtol=0,atol=1e-5)

    def test_soft_ranking_loss_gradient(self):
        scores=torch.tensor([0.1,0.2,-0.1],requires_grad=True)
        utility=torch.tensor([5.,3.,1.])
        loss=teacher_soft_loss(scores,utility)
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        self.assertTrue(bool(torch.isfinite(scores.grad).all()))
        self.assertGreater(float(scores.grad.abs().sum()),0)

if __name__=="__main__":
    unittest.main()
