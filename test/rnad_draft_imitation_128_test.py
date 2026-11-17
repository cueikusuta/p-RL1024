"""Fast independent guardrails for 128-teacher Match imitation research."""
from __future__ import annotations
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch

from scripts.rnad.train_draft_deepsets import DeepSetsDraft, FEATURE_WIDTH
from scripts.rnad.train_draft_imitation_128 import (
    BASELINE_SHA, HOLDOUT_MATCHES, cluster_bootstrap80,
    restore_old_32match_model, sharp_teacher_loss,
)

class DraftImitation128Test(unittest.TestCase):
    def test_roster_embedding_cache_exact_scores_and_gradient(self):
        torch.manual_seed(149)
        model=DeepSetsDraft()
        ref=DeepSetsDraft()
        ref.load_state_dict(model.state_dict())
        pool=torch.randn(9,FEATURE_WIDTH)
        codes=torch.tensor([[0,1,2,3,4,5],[1,2,3,4,6,7],
                            [3,2,6,4,5,8]],dtype=torch.long)
        extra=torch.randn(3,4)
        y=model(pool,codes,extra)
        # Reference to the OLD implementation without caching.
        m=ref.phi(pool[codes])
        out=ref.rho(torch.cat(
            [m.mean(1),(m*m).mean(1),m.amax(1),extra],dim=1)).squeeze(-1)
        torch.testing.assert_close(y,out,rtol=1e-5,atol=1e-6)
        y.sum().backward()
        out.sum().backward()
        for (k,a),(key,b) in zip(model.named_parameters(),
                                 ref.named_parameters()):
            self.assertEqual(k,key)
            torch.testing.assert_close(a.grad,b.grad,rtol=1e-5,atol=1e-5)

    def test_sharp_top1_teacher_loss_has_finite_nonzero_gradient(self):
        logits=torch.tensor([.2,-.7,.5,-1.],requires_grad=True)
        utility=torch.tensor([3.,8.,2.,0.])
        loss=sharp_teacher_loss(logits,utility,1)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all().item())
        self.assertLess(float(logits.grad[1]),0)
        self.assertGreater(float(logits.grad.abs().sum()),0)
        with self.assertRaises(ValueError):
            sharp_teacher_loss(logits.detach(),utility,10)

    def test_restores_frozen_strict_model_and_rejects_tampering(self):
        model=DeepSetsDraft()
        raw={"schema":"rnad-draft-deepsets-weights-v1",
            "architecture":"deepsets-soft",
            "feature_width":FEATURE_WIDTH,
            "source_sha":"e"*40,
            "weights":{k:{"shape":list(v.shape),
                          "values":v.detach().flatten().tolist()}
                        for k,v in model.state_dict().items()}}
        content=json.dumps(raw,separators=(",",":")).encode()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"model.json"
            path.write_bytes(content)
            with patch("scripts.rnad.train_draft_imitation_128.BASELINE_SHA",
                       hashlib.sha256(content).hexdigest()):
                result=restore_old_32match_model(str(path))
                self.assertFalse(result.training)
                torch.testing.assert_close(
                    result.state_dict()["rho.4.bias"],
                    model.state_dict()["rho.4.bias"])
                path.write_bytes(content+b" ")
                with self.assertRaisesRegex(ValueError,"SHA mismatch"):
                    restore_old_32match_model(str(path))

    def test_80pct_seed_cluster_boundaries_are_deterministic(self):
        b={i:{"top1":0.,"top10":.3,"regret":.5}
           for i in range(HOLDOUT_MATCHES)}
        c={i:{"top1":1.,"top10":.5,"regret":.3}
           for i in range(HOLDOUT_MATCHES)}
        seeds=list(b)
        first=cluster_bootstrap80(b,c,seeds,replicates=1000)
        second=cluster_bootstrap80(b,c,seeds,replicates=1000)
        self.assertEqual(first,second)
        self.assertAlmostEqual(first["top1"]["difference"],1.)
        self.assertEqual(first["top1"]["familywise_80pct_approximate_interval"],
                         [1.,1.])
        with self.assertRaises(ValueError):
            cluster_bootstrap80(b,c,seeds[:-1])

if __name__=="__main__":
    unittest.main()
