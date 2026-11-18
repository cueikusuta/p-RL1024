"""Wave 1 -> Wave 2 handoff fails closed on mixed generations and source SHA.

Synthetic fixtures ONLY. Live Actor/Trainer CI must additionally restore
GitHub's actual generation-one checkpoint and authoritative Match artifacts.
"""
import json
import tempfile
import unittest
from pathlib import Path

from scripts.rnad.shared_league_contract import ACTORS, SEEDS_PER_ACTOR
from scripts.rnad.shared_league_handoff import (
    _check_full_match_ids, _seal_bundle, ARTIFACT_SCHEMA,
)
from scripts.rnad.shared_league_bootstrap import MODEL_SCHEMA, POLICY_ID, digest


class HandoffContractTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.old_sha="a"*40
        self.new_sha="b"*40
        self.engine="c"*40

    def test_full_prior_generation_ids_must_match_exact_640_seeds_both_seats(self):
        prior={"generation":1,"seed_start":9900641}
        ids=[f"g1-a{a}-s{seed}-seat{seat}" for a in range(ACTORS)
             for seed in range(
                 prior["seed_start"]+a*SEEDS_PER_ACTOR,
                 prior["seed_start"]+(a+1)*SEEDS_PER_ACTOR)
             for seat in (0,1)]
        self.assertEqual(len(ids),1280)
        _check_full_match_ids(ids,prior)
        for wrong in (ids[:-1],ids[:-1]+[ids[0]],
                      ids[:-1]+["g1-a19-s9901279-seat0"],
                      ids[:-1]+["g2-a19-s9901280-seat1"]):
            with self.assertRaisesRegex(ValueError,"Match"):
                _check_full_match_ids(wrong,prior)

    def test_only_manifest_is_resigned_model_weights_stay_exact(self):
        before=self.root/"old"
        before.mkdir()
        original={}
        for k in ("draft","selection","battle","values"):
            path=before/(k+".json")
            path.write_text(json.dumps({"tensor":k,"values":[1,2,3]})+"\n")
            original[k]=digest(path)
        (before/"manifest.json").write_text(json.dumps({
            "schema":MODEL_SCHEMA,
            "generation":1,
            "policy_id":POLICY_ID,
            "source_sha":self.old_sha,
            "engine_sha":self.engine,
            "files":original,
        }))
        after=self.root/"new"
        sealed=_seal_bundle(before,after,expected_generation=1,
                             old_sha=self.old_sha,new_sha=self.new_sha,
                             engine=self.engine)
        self.assertEqual(sealed,digest(after/"manifest.json"))
        self.assertEqual(json.loads((after/"manifest.json").read_text())
                         ["source_sha"],self.new_sha)
        self.assertEqual(
            json.loads((after/"manifest.json").read_text())["files"],original)
        for name,sha in original.items():
            self.assertEqual(digest(after/(name+".json")),sha)
        with self.assertRaisesRegex(FileExistsError,"already exists"):
            _seal_bundle(before,after,expected_generation=1,
                         old_sha=self.old_sha,new_sha=self.new_sha,
                         engine=self.engine)

    def test_bad_source_or_bad_tensor_sha_cannot_be_handed_off(self):
        previous=self.root/"wrong"
        previous.mkdir()
        for name in ("draft","selection","battle","values"):
            (previous/(name+".json")).write_text('{"weight":0}')
        (previous/"manifest.json").write_text(json.dumps({
            "schema":MODEL_SCHEMA,
            "generation":1,
            "policy_id":POLICY_ID,
            "source_sha":self.old_sha,
            "engine_sha":self.engine,
            "files":{name:"0"*64 for name in
                     ("draft","selection","battle","values")}}))
        with self.assertRaisesRegex(ValueError,"frozen Head"):
            _seal_bundle(previous,self.root/"bad",
                         expected_generation=1,old_sha=self.old_sha,
                         new_sha=self.new_sha,engine=self.engine)
        with self.assertRaisesRegex(ValueError,"identity"):
            _seal_bundle(previous,self.root/"bad-generation",
                         expected_generation=2,old_sha=self.old_sha,
                         new_sha=self.new_sha,engine=self.engine)


if __name__=="__main__":
    unittest.main()
