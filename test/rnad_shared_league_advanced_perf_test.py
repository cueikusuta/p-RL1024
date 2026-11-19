"""Isolated CPU throughput safety tests (synthetic; never strength evidence)."""
import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from rnad_shared_league_contract_test import manifest, payload
from scripts.rnad.shared_league_bootstrap import digest
from scripts.rnad.shared_league_learner import (
    _cached_verified, _verified_phase_tensors,
)
from scripts.rnad.shared_league_pipelined import StrictStreamingStager


class AdvancedPerfContracts(unittest.TestCase):
    def test_strict_tensor_cache_bound_to_verified_match_and_sha(self):
        m = manifest()
        accepted = payload(m, 0)["matches"][0]
        packed = _verified_phase_tensors(accepted)
        phases = [e["phase"] for e in packed["events"]]
        self.assertEqual(phases, ["draft", "draft", "selection",
                                  "selection", "battle"])
        self.assertIsInstance(
            packed["events"][0]["phase_input"]["pool"], torch.Tensor)
        self.assertIsInstance(
            packed["events"][2]["phase_input"]["own_cat"], torch.Tensor)
        self.assertIsInstance(
            packed["events"][4]["phase_input"]["state"], torch.Tensor)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root/(accepted["match_id"]+".pt")
            torch.save(packed, target)
            entry = {"match_id":accepted["match_id"],
                     "tensor_cache_sha256":digest(target)}
            row = _cached_verified(root, entry, tensor_cache=True)
            self.assertEqual(row["match_id"], accepted["match_id"])
            self.assertEqual(row["completed"], True)
            target.write_bytes(target.read_bytes()+b"tampering")
            with self.assertRaisesRegex(ValueError,
                                         "Corrupt strict tensor cache"):
                _cached_verified(root, entry, tensor_cache=True)

    def test_out_of_order_actor_streaming_has_strict_barrier(self):
        m = manifest()
        with tempfile.TemporaryDirectory() as directory, (
                patch("scripts.rnad.shared_league_pipelined.current_git_sha",
                      return_value=m["source_sha"])), (
                patch("scripts.rnad.shared_league_pipelined.frozen_engine_sha",
                      return_value=m["engine_sha"])):
            root = Path(directory)
            (root/"bootstrap").mkdir()
            (root/"bootstrap"/"generation-manifest.json").write_text(
                json.dumps(m))
            stager = StrictStreamingStager(root/"bootstrap", root/"stage")
            with self.assertRaisesRegex(ValueError, "Incomplete 20/20"):
                stager.release()
            for actor_id in reversed(range(20)):
                path = root/("actor-"+str(actor_id)+".json.gz")
                with gzip.open(path,"wt") as out:
                    json.dump(payload(m, actor_id), out,
                              separators=(",",":"), allow_nan=False)
                stored = stager.ingest(actor_id, path)
                self.assertEqual(stored["actor_id"], actor_id)
                # An identical retried artifact is idempotent.
                self.assertEqual(stager.ingest(actor_id,path),stored)
            ready=stager.release()
            self.assertEqual(ready["accepted_complete_matches"], 1280)
            self.assertTrue(ready["one_optimizer"])
            self.assertFalse(ready["optimizer_updates_started"])
            self.assertFalse(ready["champion_promoted"])
            self.assertEqual([v["actor_id"] for v in ready["actors"]],
                             list(range(20)))
            bad = root/"stage"/"actor-0"/"actor-0.json.gz"
            bad.write_bytes(bad.read_bytes()+b"corrupt")
            with self.assertRaisesRegex(ValueError,
                                         "Staged immutable actor artifact damaged"):
                stager.release()


if __name__=="__main__":
    unittest.main()
