"""Contracts for selection-only 60-action structured teacher imitation."""
from __future__ import annotations
import copy
import unittest
import torch

from scripts.rnad.train_selection_specialist import (
    TRIPLES,TRIO_ROWS,LEAD_ROWS,SelectionSpecialist,teacher_loss,
    feature_sufficiency_audit,paired_bootstrap80,
)

class SelectionSpecialistTests(unittest.TestCase):
    def test_60_actions_have_20_unordered_trios_three_leads_each(self):
        self.assertEqual(len(TRIPLES),60)
        self.assertEqual(len(TRIO_ROWS),20)
        self.assertTrue(all(len(x)==3 for x in TRIO_ROWS))
        self.assertEqual(len(LEAD_ROWS),6)
        self.assertTrue(all(len(x)==10 for x in LEAD_ROWS))
        for rows in TRIO_ROWS:
            self.assertEqual(len({TRIPLES[j][0] for j in rows}),3)
            self.assertEqual(len({tuple(sorted(TRIPLES[j])) for j in rows}),1)

    def test_both_models_emit_legal_finite_logits_and_gradients(self):
        for structured in (False,True):
            with self.subTest(structured=structured):
                torch.manual_seed(17)
                model=SelectionSpecialist(structured=structured)
                s=torch.randn(3,96)
                a=torch.randn(3,60,48)
                out=model(s,a)
                self.assertEqual(tuple(out.shape),(3,60))
                self.assertTrue(bool(torch.isfinite(out).all()))
                utility=torch.randn(3,60)
                utility[:,4]=10
                choice=torch.tensor([4,4,4])
                for mixed in (False,True):
                    model.zero_grad(set_to_none=True)
                    loss=teacher_loss(model(s,a),choice,utility,mixed=mixed)
                    self.assertTrue(bool(torch.isfinite(loss)))
                    loss.backward()
                    self.assertGreater(sum(float(x.grad.abs().sum()) for x in model.parameters()),0)
                for tensor in (s[:,:-1],a[:,:59,:]):
                    with self.assertRaises(ValueError):
                        if tensor.shape[-1]==95:
                            model(tensor,a)
                        else:
                            model(s,tensor)

    def test_hard_only_and_hardsoft_targets_are_different(self):
        v=torch.tensor([[0.,0.,0.,0.]],requires_grad=True)
        labels=torch.tensor([2])
        util=torch.tensor([[5.,1.,6.,0.]])
        h=teacher_loss(v,labels,util,mixed=False)
        m=teacher_loss(v,labels,util,mixed=True)
        # At identical logits the losses happen to match; gradients differ.
        h.backward(retain_graph=True)
        hard_grad=v.grad.detach().clone()
        v.grad.zero_()
        m.backward()
        self.assertFalse(torch.allclose(hard_grad,v.grad))

    def test_feature_alias_audit_flags_utility_conflict(self):
        options=[[0.0]*48 for _ in range(60)]
        utils=[float(i) for i in range(60)]
        event={"phase":"selection","actions":options,
               "candidate_utilities":utils}
        out=feature_sufficiency_audit([{"seed":101,"examples":[event]}])
        self.assertEqual(out["selection_teacher_decisions"],1)
        self.assertEqual(out["decisions_with_action_feature_utility_collision"],1)
        self.assertGreater(out["trio_groups_with_indistinguishable_leads"],0)
        self.assertFalse(out["input_sufficiency_proven"])

    def test_paired_seed_bootstrap_is_clustered_and_deterministic(self):
        ids=list(range(128))
        old={i:{"top1":.25} for i in ids}
        new={i:{"top1":.40} for i in ids}
        a=paired_bootstrap80(old,new,ids)
        b=paired_bootstrap80(old,new,ids)
        self.assertEqual(a,b)
        self.assertAlmostEqual(a["per_match_cluster_top1_delta"],.15,places=6)
        self.assertAlmostEqual(a["approximate_familywise80_CI"][0],.15,places=6)

if __name__=="__main__":
    unittest.main()
