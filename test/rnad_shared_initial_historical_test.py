"""G0 initial historical opponent must map only to the SHA-locked Main."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.rnad.shared_league_collect import _historical_opponent_path


class InitialHistoricalOpponentTest(unittest.TestCase):
    def test_genesis_actor_alias_is_sha_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "actor").mkdir()
            payload = b'{"schema":"pinned","generation":0}\n'
            (root / "actor" / "manifest.json").write_bytes(payload)
            sha = hashlib.sha256(payload).hexdigest()
            manifest = {"generation":0,"main_sha256":sha}
            opponent = {"id":"integrated-initial-snapshot-v0","sha256":sha}
            self.assertEqual(_historical_opponent_path(root,manifest,opponent),
                             root / "actor")
            with self.assertRaisesRegex(ValueError,"approved registry"):
                _historical_opponent_path(root,manifest,
                    {**opponent,"sha256":"0"*64})
            with self.assertRaisesRegex(ValueError,"approved registry"):
                _historical_opponent_path(root,{**manifest,"main_sha256":"1"*64},
                                          opponent)
            with self.assertRaisesRegex(ValueError,"approved registry"):
                _historical_opponent_path(root,{**manifest,"generation":1},
                                          opponent)

    def test_later_history_requires_real_pinned_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root/"past"/"past-g0"
            folder.mkdir(parents=True)
            payload=b'{"generation":0}\n'
            (folder/"manifest.json").write_bytes(payload)
            sha=hashlib.sha256(payload).hexdigest()
            opponent={"id":"past-g0","sha256":sha}
            self.assertEqual(_historical_opponent_path(
                root,{"generation":1,"main_sha256":"0"*64},opponent),folder)
            with self.assertRaisesRegex(ValueError,"Untrusted"):
                _historical_opponent_path(
                    root,{"generation":1,"main_sha256":"0"*64},
                    {"id":"../past-g0","sha256":sha})


if __name__ == "__main__":
    unittest.main()
