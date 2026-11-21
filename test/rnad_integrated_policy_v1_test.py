"""Integrated three-expert trainable wrapper preflight (no invented learning)."""
import unittest

import torch

from scripts.rnad.integrated_policy_v1 import (
    IntegratedPolicyV1, restore_experts, TEACHER_MANIFEST_SCHEMA,
)
from scripts.rnad.train_draft_deepsets import FEATURE_WIDTH
from scripts.rnad.train_selection_specialist import TRIPLES


class IntegratedPolicyV1Test(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261118)
        vocab = {"sizes": {k: 3 for k in (
            "species", "names", "abilities", "items", "moves", "types")}}
        self.model = IntegratedPolicyV1(vocab)
        self.draft_view = {
            "pool": torch.randn(6, FEATURE_WIDTH),
            "extra": torch.randn(1, 4),
        }
        self.draft_legal = torch.tensor([[0, 1, 2, 3, 4, 5]],
                                        dtype=torch.long)
        self.selection_view = {
            "own_cat": torch.zeros(1, 6, 9, dtype=torch.long),
            "foe_cat": torch.zeros(1, 6, 3, dtype=torch.long),
            "own_num": torch.randn(1, 6, 16),
            "foe_num": torch.randn(1, 6, 6),
        }
        self.battle_view = {"state": torch.randn(1, 96)}
        self.battle_legal = torch.randn(1, 4, 48)

    def test_all_three_heads_forward_and_critic_with_one_model(self):
        dl, dv, dm = self.model(
            "draft", self.draft_view, self.draft_legal)
        sl, sv, sm = self.model(
            "selection", self.selection_view, TRIPLES)
        bl, bv, bm = self.model(
            "battle", self.battle_view, self.battle_legal,
            self.model.initial_memory())
        self.assertEqual(tuple(dl.shape), (1,))
        self.assertEqual(tuple(sl.shape), (60,))
        self.assertEqual(tuple(bl.shape), (4,))
        self.assertEqual(tuple(bm.shape), (1, 128))
        self.assertIsNone(dm)
        self.assertIsNone(sm)
        for value in (dl, sl, bl, dv, sv, bv, bm):
            self.assertTrue(torch.isfinite(value).all())

    def test_shared_learner_gradient_reaches_each_head_and_value(self):
        terms = []
        for phase, obs, actions, memory in [
            ("draft", self.draft_view, self.draft_legal, None),
            ("selection", self.selection_view, TRIPLES, None),
            ("battle", self.battle_view, self.battle_legal,
             self.model.initial_memory()),
        ]:
            logits, value, _ = self.model(phase, obs, actions, memory)
            terms.append(logits.mean() + value)
        sum(terms).backward()
        for parameter in [
            self.model.draft.phi[0].weight,
            self.model.selection.embed.own_proj[0].weight,
            self.model.battle.state_proj.weight,
            self.model.draft_value[0].weight,
            self.model.selection_value[0].weight,
        ]:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_secret_or_teacher_inputs_forbidden(self):
        bad = dict(self.draft_view, teacher_utility=1.)
        with self.assertRaisesRegex(ValueError, "privileged"):
            self.model("draft", bad, self.draft_legal)
        bad = dict(self.selection_view, opponent_private="hidden")
        with self.assertRaisesRegex(ValueError, "privileged"):
            self.model("selection", bad, TRIPLES)
        bad = dict(self.battle_view, teacher_logits=[1, 2])
        with self.assertRaisesRegex(ValueError, "privileged"):
            self.model("battle", bad, self.battle_legal,
                       self.model.initial_memory())

    def test_wrong_legal_selection_and_recurrent_state_fail_closed(self):
        bad = [*TRIPLES[:-1], TRIPLES[0]]
        with self.assertRaisesRegex(ValueError, "canonical"):
            self.model("selection", self.selection_view, bad)
        with self.assertRaisesRegex(ValueError, "recurrent"):
            self.model("battle", self.battle_view, self.battle_legal,
                       torch.zeros(2, 128))

    def test_readonly_actor_copy_is_not_separate_trained_model(self):
        actor = self.model.immutable_actor_copy()
        self.assertTrue(all(not p.requires_grad for p in actor.parameters()))
        self.assertTrue(all(p.requires_grad for p in self.model.parameters()))
        self.assertIsNot(next(self.model.parameters()), next(actor.parameters()))

    def test_restore_refuses_mismatched_source_sha_before_any_load(self):
        manifest = {
            "schema": TEACHER_MANIFEST_SCHEMA,
            "policy_id": "test-integrated",
            "sources": {
                k: "a" * 64 for k in (
                    "draft", "selection", "selection_study",
                    "battle", "battle_report")
            },
        }
        with self.assertRaisesRegex(ValueError, "SHA256"):
            restore_experts(
                self.model, manifest=manifest,
                draft_path="/path-not-found/nonexistent",
                selection_path="/path-not-found/nonexistent",
                selection_study_path="/path-not-found/nonexistent",
                battle_path="/path-not-found/nonexistent",
                battle_report_path="/path-not-found/nonexistent",
            )


if __name__ == "__main__":
    unittest.main()
