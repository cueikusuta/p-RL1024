"""Confirm filtered teacher-valid holdout cannot be forged, relabeled or silently padded."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from scripts.rnad.draft_battle_learning_512_recovery import verified_accepted_holdout
from scripts.rnad.recover_draft_battle_holdout_512 import (
    SOURCE, SCHEMA, START, TARGET, MAX_CANDIDATES, MAX_REJECTIONS)


class StrictHoldoutManifestTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.trace_dir = self.root / "traces"
        self.trace_dir.mkdir()
        self.ids = list(range(START, START+TARGET))
        self.hashes = {}
        for i in self.ids:
            file = self.trace_dir / f"teacher-{i}.json"
            file.write_bytes(b'{"safe":"dummy fixture"}')
            self.hashes[str(i)] = hashlib.sha256(file.read_bytes()).hexdigest()
        self.doc = {
            "schema": SCHEMA,
            "source_sha": SOURCE, "engine_sha": "a"*40,
            "selection_mechanism": "first_512_successful_original_source_authority_match_utility_invariant",
            "initial_seed": START, "maximum_prespecified_candidate_seeds":MAX_CANDIDATES,
            "maximum_rejections":MAX_REJECTIONS, "unseen_complete_matches": TARGET,
            "accepted_seeds":self.ids[:], "rejected_seeds":[],
            "trace_sha256_by_seed":self.hashes.copy(),
            "teacher_success_filter_applied_before_any_model_evaluation":True,
            "scorer_or_teacher_policy_altered":False,
            "unconditional_original_512_seed_population":True,
            "statistical_claim_limited_to_verified_teacher_valid_population":True,
            "league_entry_allowed":False,"champion_promoted":False
        }
        self.manifest = self.root / "manifest.json"
        self.save()

    def save(self):
        self.manifest.write_text(json.dumps(self.doc))

    def verified(self):
        with patch("scripts.rnad.draft_battle_learning_512_recovery.frozen_engine_sha",
                   return_value="a"*40):
            return verified_accepted_holdout(self.trace_dir, self.manifest)

    def test_valid_canonical_seed_list(self):
        seeds, report = self.verified()
        self.assertEqual(seeds, self.ids)
        self.assertEqual(report["rejected_seeds"], [])

    def test_rejected_seed_requires_exact_success_order(self):
        missing = START+5
        self.doc["accepted_seeds"] = [x for x in self.ids if x != missing] + [START+TARGET]
        self.doc["rejected_seeds"] = [
            {"seed":missing,"reason":"rejected_original_public_utility_invariant"}]
        self.doc["unconditional_original_512_seed_population"] = False
        self.trace_dir.joinpath(f"teacher-{missing}.json").unlink()
        self.trace_dir.joinpath(f"teacher-{START+TARGET}.json").write_bytes(b'{"safe":"replacement"}')
        del self.doc["trace_sha256_by_seed"][str(missing)]
        file = self.trace_dir / f"teacher-{START+TARGET}.json"
        self.doc["trace_sha256_by_seed"][str(START+TARGET)] = hashlib.sha256(file.read_bytes()).hexdigest()
        self.save()
        ids, _ = self.verified()
        self.assertEqual(ids[-1], START+TARGET)

    def test_forged_or_corrupt_trace_rejected(self):
        self.trace_dir.joinpath(f"teacher-{START}.json").write_text("evil")
        with self.assertRaisesRegex(ValueError, "trace bytes"):
            self.verified()

    def test_duplicate_or_out_of_order_ids_rejected(self):
        self.doc["accepted_seeds"][2] = self.doc["accepted_seeds"][1]
        self.save()
        with self.assertRaisesRegex(ValueError, "distinct"):
            self.verified()

    def test_unreported_skipped_seed_rejected(self):
        self.doc["accepted_seeds"] = self.ids[1:]+[START+TARGET]
        self.doc["trace_sha256_by_seed"][str(START+TARGET)] = "b"*64
        self.doc["unconditional_original_512_seed_population"] = False
        self.save()
        with self.assertRaisesRegex(ValueError, "deterministic"):
            self.verified()


if __name__=="__main__":
    unittest.main()
