"""The league cannot start until all three source-bound heldout Top1 gates pass."""
from __future__ import annotations

import copy
import unittest
from scripts.rnad.league_entry_gate import gate_report, verify_config

CONFIG={
    "schema":"rnad-i5-league-entry-thresholds-v1",
    "confidence_level":.80, "familywise_one_sided_alpha":.20,
    "independent_unit":"heldout_teacher_complete_match_seed",
    "min_independent_seed_clusters_per_phase":128,
    "min_accepted_teacher_decisions_per_phase":256,
    "phase_top1_targets":{
       "draft":{"threshold":.50,"comparison":"strictly_greater"},
       "selection":{"threshold":.80,"comparison":"at_least"},
       "battle":{"threshold":.80,"comparison":"at_least"}},
    "require_same_integrated_policy_id":True,
    "require_distinct_training_and_holdout_seeds":True,
    "require_complete_authority_accepted_heldout":True,
    "league_training_started":False,
    "league_champion_promoted":False,
}

def evidence(correct=2, n=128):
    return {"schema":"rnad-i5-three-phase-accepted-teacher-heldout-v1",
        "integrated_policy_id":"frozen-3phase-test-model",
        "source_sha":"a"*40,"engine_sha":"b"*64,
        "run_conclusion":"success",
        "authority_match_legality_verified":True,
        "actor_private_info_noninterference_verified":True,
        "phases":{phase:{
            "model_sha256":str(k)*64,
            "heldout_trace_sha256":"c"*64,
            "teacher_id":"aq-public-accepted-"+phase,
            "training_seeds":[111,112],
            "seed_clusters":[{
                "seed":1000+j, "accepted_decisions":2,
                "top1_correct":correct, "completed_four_win_match":True,
                "teacher_action_accepted":True}
                for j in range(n)]}
            for k,phase in enumerate(["draft","selection","battle"],1)}}

class LeagueEntryTest(unittest.TestCase):
    def test_policy_exact_user_three_thresholds(self):
        self.assertTrue(verify_config(CONFIG))
        changed=copy.deepcopy(CONFIG)
        changed["phase_top1_targets"]["selection"]["threshold"]=.70
        with self.assertRaises(ValueError):
            verify_config(changed)

    def test_missing_evidence_is_hold_not_success(self):
        result=gate_report(CONFIG,None)
        self.assertFalse(result["league_training_allowed"])

    def test_true_all_passes_only_after_independent_clusters(self):
        r=gate_report(CONFIG,evidence())
        self.assertTrue(r["league_training_allowed"])
        self.assertTrue(all(x["passed"] for x in r["phases"].values()))
        self.assertFalse(r["strength_established"])
        self.assertFalse(r["champion_promoted"])

    def test_top1_just_above_threshold_does_not_claim_statistical_pass(self):
        e=evidence()
        # 128 independent clusters, exactly half the drafts correct (weak).
        for j,x in enumerate(e["phases"]["draft"]["seed_clusters"]):
            x["top1_correct"]=2 if j%2==0 else 0
        r=gate_report(CONFIG,e)
        self.assertFalse(r["league_training_allowed"])
        self.assertFalse(r["phases"]["draft"]["passed"])

    def test_selection_alone_missing_prevents_league(self):
        e=evidence()
        for row in e["phases"]["selection"]["seed_clusters"]:
            row["top1_correct"]=1
        r=gate_report(CONFIG,e)
        self.assertFalse(r["league_training_allowed"])
        self.assertTrue(r["phases"]["draft"]["passed"])
        self.assertFalse(r["phases"]["selection"]["passed"])

    def test_duplicate_seed_training_overlap_and_fake_accept_are_rejected(self):
        e=evidence()
        e["phases"]["draft"]["seed_clusters"][1]["seed"]=1000
        with self.assertRaisesRegex(ValueError,"Duplicate"):
            gate_report(CONFIG,e)
        e=evidence()
        e["phases"]["draft"]["training_seeds"].append(1000)
        with self.assertRaisesRegex(ValueError,"seed reuse"):
            gate_report(CONFIG,e)
        e=evidence()
        e["phases"]["battle"]["seed_clusters"][0]["completed_four_win_match"]=False
        with self.assertRaisesRegex(ValueError,"incomplete"):
            gate_report(CONFIG,e)
        e=evidence()
        e["actor_private_info_noninterference_verified"]=False
        with self.assertRaisesRegex(ValueError,"Unverified"):
            gate_report(CONFIG,e)

    def test_no_qualifying_sample_means_hold(self):
        e=evidence(n=10)
        r=gate_report(CONFIG,e)
        self.assertFalse(r["league_training_allowed"])
        self.assertEqual(r["phases"]["draft"]["reason"],"too_few_independent_holdout_seeds")

if __name__=="__main__":
    unittest.main()
