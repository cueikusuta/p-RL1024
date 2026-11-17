"""A-only scaling strict independent completed Match and bootstrap gates."""
import unittest
import torch
from scripts.rnad.selection_a_1024 import (
    TRAIN_BLOCKS, DEV_BLOCK, EPOCHS, SEEDS, N_MATCHES,
)
from scripts.rnad.selection_a_1024_confirm import (
    HOLDOUT_START, HOLDOUT_MATCHES, cluster_delta,
)
from scripts.rnad.selection_ab import DecomposedTeacher, objective
from test.rnad_selection_ab_test import fixture


class A1024Contracts(unittest.TestCase):
    def test_seed_blocks_are_disjoint_and_not_tiny(self):
        blocks=[set(range(start,start+n))
                for start,n in TRAIN_BLOCKS]
        blocks.append(set(range(DEV_BLOCK[0],DEV_BLOCK[0]+DEV_BLOCK[1])))
        blocks.append(set(range(HOLDOUT_START,HOLDOUT_START+HOLDOUT_MATCHES)))
        self.assertEqual(sum(map(len,blocks[:3])),N_MATCHES)
        self.assertEqual(HOLDOUT_MATCHES,512)
        self.assertEqual(len(SEEDS),3)
        self.assertEqual(EPOCHS,(20,40))
        for i in range(len(blocks)):
            for j in range(i+1,len(blocks)):
                self.assertFalse(blocks[i]&blocks[j])

    def test_complete_match_seed_cluster_bootstrap(self):
        before={x:.25 for x in range(HOLDOUT_MATCHES)}
        after={x:.3 for x in range(HOLDOUT_MATCHES)}
        result=cluster_delta(before,after)
        self.assertEqual(result["independent_complete_Match_seed_clusters"],512)
        self.assertAlmostEqual(result["paired_Match_mean_delta"],.05)
        self.assertAlmostEqual(result["approximate_two_sided_95pct_ci"][0],.05)
        with self.assertRaisesRegex(ValueError,"512"):
            cluster_delta(dict(list(before.items())[:-1]),after)

    def test_A_only_forward_uses_public_actor_not_teacher_target(self):
        torch.set_num_threads(1)
        example,vocab=fixture()
        model=DecomposedTeacher(vocab)
        tensors={
            "own_cat":torch.tensor([example["actor"]["own_cats"]]),
            "foe_cat":torch.tensor([example["actor"]["foe_cats"]]),
            "own_num":torch.tensor([example["actor"]["own_numeric"]],
                                    dtype=torch.float32),
            "foe_num":torch.tensor([example["actor"]["foe_numeric"]],
                                    dtype=torch.float32),
        }
        logits,core,lead=model(tensors)
        self.assertEqual(tuple(logits.shape),(1,60))
        self.assertEqual(tuple(core.shape),(1,20))
        self.assertEqual(tuple(lead.shape),(1,6))
        self.assertTrue(torch.isfinite(logits).all().item())
        loss=objective(model,[(1,example)],"A")
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum().item()>0
                            for p in model.parameters()))


if __name__=="__main__":
    unittest.main()
