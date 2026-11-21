"""Shared IntegratedPolicyV1 exact full-state checkpoint restart contract.

These are synthetic deterministic optimizer checkpoints, NOT accepted game
trajectories and never evidence of real Match learning or strength.
"""
import copy
import json
import random
import tempfile
import unittest
from pathlib import Path

import torch

from scripts.rnad.integrated_policy_v1 import IntegratedPolicyV1
from scripts.rnad.shared_league_bootstrap import POLICY_ID, digest
from scripts.rnad.shared_league_learner import _models, _atomic_checkpoint, SCHEMA


class SharedLearnerResumeTest(unittest.TestCase):
    def test_full_state_optimizer_rng_model_target_and_magnet_bitwise_replay(self):
        with tempfile.TemporaryDirectory() as dirname:
            root=Path(dirname)
            actor=root/"actor"
            actor.mkdir()
            vocab={"sizes":{key:3 for key in
                ("species","names","abilities","items","moves","types")}}
            (actor/"selection.json").write_text(json.dumps({
                "public_actor_vocabulary":vocab}))
            torch.manual_seed(22026)
            random.seed(22026)
            net=IntegratedPolicyV1(vocab)
            target=copy.deepcopy(net)
            magnet=copy.deepcopy(net)
            optimizer=torch.optim.Adam(net.parameters(),lr=3e-5)
            scheduler=torch.optim.lr_scheduler.LambdaLR(
                optimizer,lr_lambda=lambda _:1.0)

            def step(model,opt,sch):
                opt.zero_grad(set_to_none=True)
                loss=sum((p*p).mean() for p in model.parameters())
                loss.backward()
                opt.step()
                sch.step()

            step(net,optimizer,scheduler)
            checkpoint={
                "schema":SCHEMA,
                "policy_id":POLICY_ID,
                "generation":0, "step":1,
                "model":net.state_dict(), "target":target.state_dict(),
                "magnet":magnet.state_dict(),
                "optimizer":optimizer.state_dict(),
                "scheduler":scheduler.state_dict(),
                "torch_rng":torch.get_rng_state(),
                "python_rng":random.getstate(),
                "training_match_ids":["test-synthetic-not-authoritative"],
                "is_champion":False,
            }
            path=root/"checkpoint-working.pt"
            _atomic_checkpoint(path,checkpoint)
            index=json.loads(path.with_suffix(".json").read_text())
            self.assertEqual(digest(path),index["sha256"])
            restored=torch.load(path,map_location="cpu",weights_only=True)
            self.assertEqual(restored["optimizer"]["param_groups"],
                             checkpoint["optimizer"]["param_groups"])
            self.assertEqual(restored["training_match_ids"],
                             ["test-synthetic-not-authoritative"])

            reloaded,r_target,r_magnet,ro,rs=_models(root,restored)
            for p,q in zip(target.parameters(),r_target.parameters()):
                self.assertTrue(torch.equal(p,q))
            for p,q in zip(magnet.parameters(),r_magnet.parameters()):
                self.assertTrue(torch.equal(p,q))
            for p,q in zip(net.parameters(),reloaded.parameters()):
                self.assertTrue(torch.equal(p,q))
            self.assertTrue(torch.equal(torch.get_rng_state(),
                                        checkpoint["torch_rng"]))
            self.assertEqual(random.getstate(),checkpoint["python_rng"])
            step(net,optimizer,scheduler)
            step(reloaded,ro,rs)
            for p,q in zip(net.parameters(),reloaded.parameters()):
                self.assertTrue(torch.equal(p,q))
            self.assertEqual(optimizer.state_dict()["param_groups"],
                             ro.state_dict()["param_groups"])
            self.assertEqual(scheduler.state_dict(),rs.state_dict())


if __name__=="__main__":
    unittest.main()
