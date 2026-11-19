"""Staging equivalence and rejection tests; synthetic data is NEVER training evidence."""
import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from rnad_shared_league_contract_test import manifest, payload
from scripts.rnad.shared_league_staging import (
    stage_verified_actors, staging_config,
)


class SharedLearnerStagingTest(unittest.TestCase):
    def _actors(self, root):
        m = manifest()
        paths = []
        for i in range(20):
            directory = root / f"actor-{i}"
            directory.mkdir()
            file = directory / f"actor-{i}.json.gz"
            with gzip.open(file, "wt") as sink:
                json.dump(payload(m, i), sink, separators=(",", ":"),
                          allow_nan=False)
            paths.append(file)
        return m, paths

    def test_all_1280_outputs_identical_under_parallel_fast_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            m, paths = self._actors(root)
            a = root / "reference"
            b = root / "fast-two"
            c = root / "fast-four"
            first = stage_verified_actors(
                m, paths, a, workers=1, fast_json=False,
                return_metrics=True)
            for workers, dest in ((2, b), (4, c)):
                candidate = stage_verified_actors(
                    m, paths, dest, workers=workers, fast_json=True,
                    return_metrics=True)
                self.assertEqual(first[0], candidate[0])
                self.assertEqual(first[1], candidate[1])
                self.assertEqual(first[2], candidate[2])
                self.assertEqual(candidate[3]["accepted_matches"], 1280)
                self.assertEqual(candidate[3]["workers"], workers)
                for row in first[1]:
                    name = row["match_id"] + ".json"
                    self.assertEqual(
                        hashlib.sha256((a / name).read_bytes()).digest(),
                        hashlib.sha256((dest / name).read_bytes()).digest(),
                        name)

    def test_corrupt_and_missing_shards_fail_before_optimizer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            m, paths = self._actors(root)
            with self.assertRaisesRegex(ValueError, "20/20"):
                stage_verified_actors(m, paths[:-1], root / "missing")
            with gzip.open(paths[0], "wt") as target:
                json.dump(payload(m, 1), target, separators=(",", ":"))
            for workers in (1, 2):
                with self.assertRaisesRegex(ValueError, "Actor artifact ID"):
                    stage_verified_actors(
                        m, paths, root / f"bad-{workers}",
                        workers=workers, fast_json=True)

    def test_invalid_staging_configuration_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "RNAD_PREP_WORKERS"):
            stage_verified_actors(manifest(), [], Path("."), workers=3)
        from unittest.mock import patch
        with patch.dict("os.environ", {"RNAD_PREP_WORKERS": "20"}):
            with self.assertRaisesRegex(ValueError, "Invalid bounded"):
                staging_config()
        with patch.dict("os.environ", {"RNAD_PREP_FAST_JSON": "bad"}):
            with self.assertRaisesRegex(ValueError, "Invalid bounded"):
                staging_config()


if __name__ == "__main__":
    unittest.main()
